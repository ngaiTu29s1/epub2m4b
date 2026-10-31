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
    """

    def __init__(
        self,
        audio_prompt: Path | None = None,
        ckpt_dir: Path | None = None,
        device: str = 'cuda',
        language: str = 'vi',
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 1.0,
        repetition_penalty: float = 2.0,
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

    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if not re.search(r'\w', text):
            return _write_silence(out_wav)

        model = get_viterbox(self.ckpt_dir, self.device)
        audio = model.generate(
            text,
            language=self.language,
            audio_prompt=str(self.audio_prompt),
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )
        wav = audio.squeeze(0).detach().float().cpu().numpy()
        return _write_mono_24k_pcm16(out_wav, wav, model.sr)


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
