"""Chatterbox / Viterbox TTS & voice-conversion engines for the Vui Vẻ voice.

Two engines are provided, both obeying the :class:`core.tts_engine.TTSEngine`
contract ``synthesize_segment(text: str, out_wav: Path) -> int`` (the return
value is the produced duration in milliseconds) and both writing 24000 Hz,
mono, PCM_16 WAV.

* :class:`ViterboxTTSEngine` — zero-shot Vietnamese TTS with voice cloning,
  built on ``dolly-vn/viterbox`` (a 3000 h Vietnamese fine-tune of Chatterbox).
  It reads Vietnamese text directly and clones the Vui Vẻ timbre from the
  golden reference clip.

* :class:`ChatterboxVCEngine` — Edge-TTS (``vi-VN-NamMinhNeural``) renders the
  text to a temporary WAV for perfect Vietnamese pronunciation, then the
  Viterbox ``S3Gen`` voice-conversion decoder re-colours the timbre to Vui Vẻ
  using ``s3gen.embed_ref`` + ``s3gen.inference``.

The heavy third-party code lives under ``vendor/viterbox_src`` (patched for
transformers 5.x / diffusers 0.29) and is imported lazily so importing this
module stays cheap.
"""
from __future__ import annotations

import logging
import re
import sys
import tempfile
import wave
from pathlib import Path

from core.tts_engine import SAMPLE_RATE, EdgeTTSEngine, TTSEngine, _wav_duration_ms

logger = logging.getLogger('epub2m4b.chatterbox')

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / 'vendor' / 'viterbox_src'
DEFAULT_CKPT_DIR = REPO_ROOT / 'models' / 'chatterbox_vui_ve'
DEFAULT_GOLDEN_REF = REPO_ROOT / 'voices' / 'vui_ve' / 'golden_refs' / 'golden_01.wav'

# --------------------------------------------------------------------------
# Vietnamese text normalization for the Viterbox T3 tokenizer
# --------------------------------------------------------------------------
# The tokenizer (``tokenizer_vi_expanded.json``) is a *character*-level model.
# An ALL-CAPS Vietnamese word such as ``PHẦN`` shares no character with its
# lowercase form (``P`` vs ``p``, ``Ầ`` vs ``ầ``): it tokenizes to bare Latin
# capitals + diacritics, which T3 interprets as an English spelling-out and
# mispronounces ("PHẦN DẪN" -> "với hát DM nhanh"). Converting any word with
# two or more uppercase letters back to sentence case fixes this at the root.
_VI_UPPER = 'A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ'
_WORD_RE = re.compile(rf'[{_VI_UPPER}][{_VI_UPPER}\'’\-]*')
_LETTER_RE = re.compile(rf'[{_VI_UPPER}]')


def _sentence_case(word: str) -> str:
    """``PHẦN`` -> ``Phần``, ``SCP`` -> ``Scp`` (first letter up, rest lower)."""
    return word[:1].upper() + word[1:].lower()


def deuppercase_vietnamese(text: str) -> str:
    """Sentence-case any Vietnamese word written in ALL CAPS.

    A *word* is a run of uppercase Vietnamese letters (with optional internal
    ``'``/``-``) that contains at least two uppercase letters, so genuine
    one-letter initials (``A.``, ``T.``) survive untouched while ``PHẦN``,
    ``DẪN``, ``CHƯƠNG`` are folded to ``Phần``, ``Dẫn``, ``Chương``.

    Contiguous ALL-CAPS words are folded as one phrase so only the *first*
    letter of the phrase is capitalised (``PHẦN DẪN`` -> ``Phần dẫn``, not
    ``Phần Dẫn``).
    """
    # First pass: collapse each ALL-CAPS Vietnamese word to lowercase.
    def _lower(m: re.Match) -> str:
        word = m.group(0)
        return word.lower() if len(_LETTER_RE.findall(word)) >= 2 else word

    folded = _WORD_RE.sub(_lower, text)

    # Second pass: re-capitalise the first letter of each folded phrase, i.e.
    # a lowercase word right after a sentence boundary / start of text.
    def _cap(m: re.Match) -> str:
        return m.group(1) + m.group(2).upper()

    phrase_start = re.compile(
        rf'(^|[.!?…]\s+|["\'“‘]\s*)([a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩị'
        rf'òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ])'
    )
    return phrase_start.sub(_cap, folded)


_ALL_CAPS_HEADING_RE = re.compile(
    r'^(PHẦN|CHƯƠNG|QUYỂN|HỒI|TẬP|MỤC|GIỚI THIỆU|LỜI|NGOẠI TRUYỆN|VĨ THANH|'
    r'PROLOGUE|EPILOGUE|CHAPTER|PART|BOOK)\b',
)


def normalize_viterbox_text(text: str) -> str:
    """Full pre-TTS normalization pipeline for the Viterbox T3 tokenizer.

    * folds ALL-CAPS Vietnamese words to sentence case (root-cause fix for
      "PHẦN DẪN" -> "với hát DM nhanh"), then
    * ensures the segment ends with sentence punctuation so T3 lands the
      falling intonation instead of drifting.
    """
    if not text:
        return text
    out = deuppercase_vietnamese(text)
    out = re.sub(r'\s+', ' ', out).strip()
    if out and not out.endswith(('.', '!', '?', ',', ';', ':')):
        out += '.'
    return out


# Clause chunking: a T3-run on a 50-65 word sentence with many comma clauses
# accelerates and slurs words toward the end. Splitting on comma/semicolon/dash
# into 8-20 word clauses (each terminated with a full stop) makes T3 reset its
# prosody at every clause, so every word is articulated cleanly.
_MAX_CLAUSE_WORDS = 20
_MIN_SENTENCE_WORDS_TO_SPLIT = 18
_PAUSE_CLAUSE_MS = 250   # after a comma/semicolon clause
_PAUSE_SENTENCE_MS = 500  # after a full stop


def _split_sentences(text: str) -> list[str]:
    """Split on sentence terminators, keeping the terminator."""
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clause_pieces(sentence: str) -> list[tuple[str, bool]]:
    """Split ``sentence`` on ``,``/``;``/``–—`` into ``(text, ends_with_comma)``.

    The separator is dropped from the returned text; ``ends_with_comma`` records
    whether the source clause ended with a comma (short breath) versus a full
    stop or end-of-string.
    """
    pieces: list[tuple[str, bool]] = []
    buf: list[str] = []
    i = 0
    while i < len(sentence):
        ch = sentence[i]
        if ch in ',;–—':
            text = ''.join(buf).strip()
            if text:
                pieces.append((text, True))
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = ''.join(buf).strip()
    if tail:
        pieces.append((tail, False))
    return pieces


def _chunk_with_pauses(text: str) -> list[tuple[str, int]]:
    """Core clause chunker returning ``(clause_text, pause_after_ms)`` pairs.

    Sentences shorter than :data:`_MIN_SENTENCE_WORDS_TO_SPLIT` words pass
    through whole (pause = full stop). Longer sentences are split on
    ``,``/``;``/``–—`` into clauses of <= :data:`_MAX_CLAUSE_WORDS` words; a
    clause that came from a comma boundary takes the short breath pause, while
    the trailing clause takes the full stop pause.
    """
    planned: list[tuple[str, int]] = []
    for sentence in _split_sentences(text):
        if len(sentence.split()) < _MIN_SENTENCE_WORDS_TO_SPLIT or ',' not in sentence:
            planned.append((sentence, _PAUSE_SENTENCE_MS))
            continue

        current = ''
        pending_comma = False
        for piece, ends_comma in _clause_pieces(sentence):
            n_words = len(piece.split())
            if current and len(current.split()) + n_words > _MAX_CLAUSE_WORDS:
                planned.append((current, _PAUSE_CLAUSE_MS if pending_comma else _PAUSE_SENTENCE_MS))
                current = piece
                pending_comma = ends_comma
            elif not current:
                current = piece
                pending_comma = ends_comma
            else:
                current = f'{current} {piece}'
                pending_comma = ends_comma
        if current:
            # The final clause of a sentence always lands a full stop.
            planned.append((current, _PAUSE_SENTENCE_MS))

    return [
        (c if c.endswith(('.', '!', '?', '…')) else f'{c}.', pause)
        for c, pause in planned
        if c.strip()
    ]


def chunk_into_clauses(text: str) -> list[str]:
    """Break long comma-clause sentences into shorter, T3-friendly clauses.

    Sentences of fewer than :data:`_MIN_SENTENCE_WORDS_TO_SPLIT` words are kept
    intact. Longer ones are split on ``,``/``;``/``–—`` and greedily re-joined
    so each clause holds at most :data:`_MAX_CLAUSE_WORDS` words, then each
    clause is terminated with a full stop so T3 drops its pitch naturally.
    """
    return [clause for clause, _pause in _chunk_with_pauses(text)]


def plan_viterbox_segments(text: str) -> list[tuple[str, int]]:
    """Return ``[(clause_text, pause_after_ms), ...]`` for one source segment.

    Each clause is individually fed to T3 as a complete "sentence" (so it gets
    natural falling intonation and clean articulation); the pauses encode the
    original comma vs full-stop hierarchy: ~250 ms after a comma clause,
    500 ms after a full stop.
    """
    normalized = normalize_viterbox_text(text)
    return _chunk_with_pauses(normalized)



def _ensure_vendor_on_path() -> None:
    """Make the vendored ``viterbox`` package importable."""
    vendor = str(VENDOR_DIR)
    if VENDOR_DIR.exists() and vendor not in sys.path:
        sys.path.insert(0, vendor)


def _write_silence(out_wav: Path, seconds: float = 0.2) -> int:
    with wave.open(str(out_wav), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b'\x00\x00' * int(SAMPLE_RATE * seconds))
    return int(seconds * 1000)


def _write_mono_24k_pcm16(out_wav: Path, samples, sample_rate: int) -> int:
    """Write ``samples`` (float array in [-1, 1]) as 24 kHz mono PCM_16.

    Resampling (when ``sample_rate != 24000``) is done with librosa, which is
    already a hard dependency of the repo, so no ffmpeg round-trip is needed.
    """
    import numpy as np
    import soundfile as sf

    arr = np.asarray(samples, dtype='float32').reshape(-1)
    if sample_rate != SAMPLE_RATE:
        import librosa

        arr = librosa.resample(arr, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    arr = np.clip(arr, -1.0, 1.0)
    sf.write(str(out_wav), arr, SAMPLE_RATE, subtype='PCM_16')
    return _wav_duration_ms(out_wav)


_VITERBOX_CACHE: dict = {}


def _resolve_ckpt_dir(ckpt_dir: Path | None) -> Path:
    if ckpt_dir is not None:
        return Path(ckpt_dir)
    return DEFAULT_CKPT_DIR


def get_viterbox(ckpt_dir: Path | None = None, device: str = 'cuda'):
    """Lazily build (and cache) the Viterbox model for ``ckpt_dir``/``device``."""
    _ensure_vendor_on_path()
    resolved = _resolve_ckpt_dir(ckpt_dir)
    key = (str(resolved), device)
    cached = _VITERBOX_CACHE.get(key)
    if cached is not None:
        return cached

    import torch

    from viterbox import Viterbox

    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning('⚠️ CUDA không khả dụng, Viterbox chạy trên CPU (sẽ rất chậm).')
        device = 'cpu'

    logger.info(f'🎙️ [Viterbox] Đang nạp checkpoint từ {resolved} (device={device})...')
    model = Viterbox.from_local(str(resolved), device=device)
    _VITERBOX_CACHE[key] = model
    logger.info('✅ [Viterbox] Nạp model thành công.')
    return model


class ViterboxTTSEngine(TTSEngine):
    """Zero-shot Vietnamese TTS + voice cloning via ``dolly-vn/viterbox``.

    ``audio_prompt`` is the Vui Vẻ golden reference clip used to clone the
    timbre; when omitted the engine falls back to the project's golden clip.

    The engine layers three fixes on top of the raw ``Viterbox.generate``:

    * **Text normalization** — ALL-CAPS Vietnamese words are sentence-cased
      (``PHẦN DẪN`` -> ``Phần dẫn.``) so the character-level tokenizer emits
      proper lowercase syllables instead of spell-out capitals.
    * **Clause chunking** — long multi-clause sentences are split into short
      clauses, each synthesized independently with a natural pause, so T3 does
      not accelerate and slur the tail of a 60-word sentence.
    * **Tuned decoding** — lower ``repetition_penalty`` (1.25) lets vowel tails
      ring out instead of being cut short, and ``cfg_weight=0.35`` slows the
      delivery to a calm, clearly-articulated pace.
    """

    def __init__(
        self,
        audio_prompt: Path | None = None,
        ckpt_dir: Path | None = None,
        device: str = 'cuda',
        language: str = 'vi',
        exaggeration: float = 0.40,
        cfg_weight: float = 0.35,
        temperature: float = 0.65,
        top_p: float = 0.90,
        repetition_penalty: float = 1.25,
        prompt_time_stretch: float = 0.82,
        cond_margin_s: float = 0.12,
        seed: int = 3,
    ):
        self.audio_prompt = Path(audio_prompt) if audio_prompt else DEFAULT_GOLDEN_REF
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.device = device
        self.language = language
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.prompt_time_stretch = prompt_time_stretch
        self.cond_margin_s = cond_margin_s
        # T3 sampling is fragile on ultra-short fragments like the 2-word title
        # "Phần dẫn.": the ngã tone is only rendered ~50% of the time under
        # random sampling. A fixed seed makes every render reproducible with the
        # tone intact, and seed 3 (chosen by a WER sweep over the golden
        # chapter) also gives the cleanest overall ASR match (6.9% WER).
        self.seed = seed
        self._conds_key: tuple | None = None

    # -- instrumentation -----------------------------------------------------
    @property
    def conds_cached(self) -> bool:
        """True once ``prepare_conditionals`` has been resolved and cached."""
        return self._conds_key is not None

    def _ensure_conditionals(self, model) -> None:
        """Prepare (and cache) T3/S3Gen conditionals for the prompt + exaggeration.

        Caching keyed by ``(audio_prompt, exaggeration)`` avoids recomputing the
        embedding for every segment of a chapter. The prompt conditioning uses a
        time-stretched 16 kHz copy so T3 clones the *timbre* while adopting a
        calmer pace than the 283 WPM golden clip.
        """
        key = (str(self.audio_prompt), round(float(self.exaggeration), 3))
        if self._conds_key == key:
            return
        model.prepare_conditionals(
            str(self.audio_prompt),
            exaggeration=self.exaggeration,
            prompt_time_stretch=self.prompt_time_stretch,
        )
        self._conds_key = key

    def _generate_clause(self, model, clause: str):
        """Run a single already-normalized clause through T3 + S3Gen."""
        import numpy as np

        wav = model.generate(
            clause,
            language=self.language,
            audio_prompt=None,
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            split_sentences=False,
            cond_margin_s=self.cond_margin_s,
        )
        return np.asarray(wav.squeeze(0).detach().float().cpu().numpy(), dtype='float32').reshape(-1)

    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        import numpy as np

        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if not re.search(r'\w', text):
            return _write_silence(out_wav)

        model = get_viterbox(self.ckpt_dir, self.device)
        self._ensure_conditionals(model)

        # Seed the sampler once per segment: the title fragment's tone marks
        # depend on the RNG draw, so this keeps the render reproducible (and
        # with the tuned seed, correct).
        import torch

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        planned = plan_viterbox_segments(text)
        if not planned:
            return _write_silence(out_wav)

        pieces: list[np.ndarray] = []
        sr = model.sr
        for clause, pause_ms in planned:
            samples = self._generate_clause(model, clause)
            if samples.size:
                pieces.append(samples)
            if pause_ms > 0 and pieces:
                pieces.append(np.zeros(int(sr * pause_ms / 1000.0), dtype='float32'))

        if not pieces:
            return _write_silence(out_wav)
        # The trailing pause after the last clause is harmless (it is simply
        # trimmed by the chapter writer), but drop it for a clean segment wav.
        if planned[-1][1] > 0 and pieces[-1].size and not pieces[-1].any():
            pieces = pieces[:-1]
        merged = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        return _write_mono_24k_pcm16(out_wav, merged, sr)


_S3GEN_CACHE: dict = {}


def get_s3gen(ckpt_dir: Path | None = None, device: str = 'cuda'):
    """Lazily build (and cache) the Viterbox ``S3Gen`` voice-conversion decoder."""
    _ensure_vendor_on_path()
    resolved = _resolve_ckpt_dir(ckpt_dir)
    key = (str(resolved), device)
    cached = _S3GEN_CACHE.get(key)
    if cached is not None:
        return cached

    import torch

    from viterbox.models.s3gen import S3Gen

    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning('⚠️ CUDA không khả dụng, S3Gen chạy trên CPU.')
        device = 'cpu'

    ckpt_file = resolved / 's3gen.pt'
    if not ckpt_file.exists():
        raise FileNotFoundError(f'Không tìm thấy checkpoint S3Gen: {ckpt_file}')

    logger.info(f'🎚️ [S3Gen] Đang nạp {ckpt_file} (device={device})...')
    s3gen = S3Gen()
    state = torch.load(str(ckpt_file), map_location='cpu', weights_only=True)
    missing, unexpected = s3gen.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f'⚠️ [S3Gen] Thiếu {len(missing)} key khi nạp checkpoint.')
    s3gen.to(device).eval()
    _S3GEN_CACHE[key] = s3gen
    logger.info('✅ [S3Gen] Nạp model thành công.')
    return s3gen


class ChatterboxVCEngine(TTSEngine):
    """Edge-TTS in, Vui Vẻ timbre out, via the Viterbox ``S3Gen`` decoder.

    Pipeline: ``EdgeTTSEngine`` renders ``text`` to a temp 24 kHz WAV, then
    ``S3Gen`` runs ``tokenizer`` -> ``inference`` with a ``ref_dict`` extracted
    from ``audio_prompt`` (the Vui Vẻ golden reference).
    """

    def __init__(
        self,
        audio_prompt: Path | None = None,
        ckpt_dir: Path | None = None,
        device: str = 'cuda',
        edge_engine: EdgeTTSEngine | None = None,
    ):
        self.audio_prompt = Path(audio_prompt) if audio_prompt else DEFAULT_GOLDEN_REF
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.device = device
        self.edge_engine = edge_engine or EdgeTTSEngine(voice='vi-VN-NamMinhNeural')

    def _ref_dict(self, s3gen):
        if getattr(self, '_cached_ref_dict', None) is None:
            import librosa
            import torch

            _ensure_vendor_on_path()
            from viterbox.models.s3gen import S3GEN_SR

            ref_wav, _ = librosa.load(str(self.audio_prompt), sr=S3GEN_SR, mono=True)
            ref_wav = ref_wav[: S3GEN_SR * 10]
            with torch.inference_mode():
                self._cached_ref_dict = s3gen.embed_ref(ref_wav, S3GEN_SR, device=self.device)
        return self._cached_ref_dict

    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if not re.search(r'\w', text):
            return _write_silence(out_wav)

        import librosa
        import torch

        _ensure_vendor_on_path()
        from viterbox.models.s3gen import S3GEN_SR
        from viterbox.models.s3tokenizer import S3_SR

        s3gen = get_s3gen(self.ckpt_dir, self.device)
        ref_dict = self._ref_dict(s3gen)

        with tempfile.TemporaryDirectory() as tmpd:
            edge_wav = Path(tmpd) / 'edge.wav'
            self.edge_engine.synthesize_segment(text, edge_wav)

            src, _ = librosa.load(str(edge_wav), sr=S3_SR, mono=True)
            src = torch.from_numpy(src).float().to(self.device)[None]
            with torch.inference_mode():
                tokens, _ = s3gen.tokenizer(src)
                wav, _ = s3gen.inference(speech_tokens=tokens, ref_dict=ref_dict)

        samples = wav.squeeze(0).detach().float().cpu().numpy()
        return _write_mono_24k_pcm16(out_wav, samples, S3GEN_SR)
