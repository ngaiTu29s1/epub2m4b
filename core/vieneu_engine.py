"""VieNeu-TTS engine (v3.8.3) — preset voices, classic & instant clone.

Wraps the ``vieneu`` package behind the project's ``TTSEngine`` interface.

Three voice modes
-----------------
``preset``
    One of VieNeu-TTS's 25 bundled voices (``Tuyên``, ``Bình``, ``Xuân Vĩnh``…).
``classic``
    The "Nguyễn Ngọc Ngạn" style. Loaded as an instant clone from a bundled /
    user reference WAV, because the upstream fine-tune
    (``nguyen-brat/nguyen-ngoc-ngan-vieneu-tts-fine-tune``) is a v2 backbone
    whose codec repo (``neuphonic/neucodec``) is gated on HF and unreachable
    without an authenticated token.
``clone``
    Instant zero-shot clone from any reference WAV (e.g. the harvested Vui Vẻ
    ``voices/vui_ve/golden_refs/golden_01.wav``, or a user upload).

All audio is resampled to the project's ``SAMPLE_RATE`` mono 16-bit PCM WAV.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.tts_engine import SAMPLE_RATE, TTSEngine, _get_ffmpeg_bin, _wav_duration_ms

logger = logging.getLogger('epub2m4b.vieneu')


def _upstream_phonemize():
    """``phonemize_text_with_emotions`` từ vieneu_utils (import lười)."""
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    return phonemize_text_with_emotions


def _upstream_chunker():
    """``normalize_to_chunks_v3_with_gaps`` từ vieneu_utils (import lười)."""
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps

    return normalize_to_chunks_v3_with_gaps


def _upstream_join():
    """``join_audio_chunks`` + ``gaps_to_silence`` — cùng logic nối của ``infer``."""
    from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks

    return join_audio_chunks, gaps_to_silence

# nguyen-brat/nguyen-ngoc-ngan-vieneu-tts-fine-tune: a v3-Turbo fine-tune whose
# weights live at the repo root and whose bundled ``tokenizer.json`` is broken,
# so it must be loaded with the *base* repo's tokenizer.
NGAN_FINE_TUNE_REPO = 'nguyen-brat/nguyen-ngoc-ngan-vieneu-tts-fine-tune'
BASE_V3_TURBO_REPO = 'pnnbao-ump/VieNeu-TTS-v3-Turbo'

# ---------------------------------------------------------------------------
# Voice catalogue
# ---------------------------------------------------------------------------
# Friendly aliases the project spec asks for ("Tuyên", "Bình", "Sơn"…) mapped
# onto the canonical preset names shipped in voices_v3_turbo.json.
PRESET_ALIASES: dict[str, str] = {
    'tuyên': 'Phạm Tuyên',
    'tuyen': 'Phạm Tuyên',
    'phạm tuyên': 'Phạm Tuyên',
    'bình': 'Thanh Bình',
    'binh': 'Thanh Bình',
    'thanh bình': 'Thanh Bình',
    'xuân vĩnh': 'Xuân Vĩnh',
    'xuan vinh': 'Xuân Vĩnh',
    'sơn': 'Quang Sơn',
    'son': 'Quang Sơn',
    'quang sơn': 'Quang Sơn',
    'thái sơn': 'Thái Sơn',
    'hải đăng': 'Hải Đăng',
    'hai dang': 'Hải Đăng',
    'adam': 'Adam bựa',
    'adam bựa': 'Adam bựa',
    'trúc ly': 'Trúc Ly',
    'thiện minh': 'Thiện Minh',
    'mai anh': 'Mai Anh',
    'thùy dung': 'Thùy Dung',
    'ngọc huyền': 'Ngọc Huyền',
    'ngọc trân': 'Ngọc Trân',
    'minh đức': 'Minh Đức',
    'quốc tuấn': 'Quốc Tuấn',
    'quỳnh anh': 'Quỳnh Anh',
    'ngọc linh': 'Ngọc Linh',
    'đoan trang': 'Đoan Trang',
    'thục đoan': 'Thục Đoan',
    'minh triết': 'Minh Triết',
    'mỹ duyên': 'Mỹ Duyên',
    'đức trí': 'Đức Trí',
    'kim thanh': 'Kim Thanh',
    'thiền tâm đức': 'Thiền Tâm Đức',
}

# Fallback catalogue when the ``vieneu`` package is not importable (tests, or a
# machine where the model was never downloaded).
FALLBACK_PRESET_VOICES: tuple[str, ...] = (
    'Adam bựa', 'Trúc Ly', 'Thiện Minh', 'Mai Anh', 'Hải Đăng', 'Thùy Dung',
    'Thiền Tâm Đức', 'Ngọc Huyền', 'Quang Sơn', 'Ngọc Trân', 'Minh Đức',
    'Phạm Tuyên', 'Thái Sơn', 'Xuân Vĩnh', 'Thanh Bình', 'Ngọc Linh',
    'Đoan Trang', 'Thục Đoan', 'Minh Triết', 'Mỹ Duyên', 'Quỳnh Anh',
    'Đức Trí', 'Kim Thanh', 'Adam', 'Quốc Tuấn',
)

DEFAULT_PRESET_VOICE = 'Hải Đăng'

# Số chunk gộp mỗi forward của VieNeu-TTS v3 Turbo trên CUDA (CUDA Graph
# batching). 4 chunk vừa 8GB VRAM nhưng vẫn đạt ~2–3× tốc độ so với batch 1.
DEFAULT_BATCH_SIZE_CUDA = 4

# Tham số sampling mặc định của ``Vieneu.infer``. ``_infer_chunks`` đọc
# ``sampling["max_new_frames"]`` trực tiếp (qua ``_cap_frames``) nên KHÔNG được
# truyền dict rỗng — phải dùng đủ bộ giá trị này.
DEFAULT_SAMPLING: dict = {
    'temperature': 0.8,
    'top_k': 25,
    'top_p': 0.95,
    'max_new_frames': 300,
    'repetition_penalty': 1.2,
}

DEFAULT_CLASSIC_NGAN_REF_WAV = (
    Path(__file__).resolve().parent.parent
    / 'voices'
    / 'nguyen_ngoc_ngan'
    / 'golden_refs'
    / 'ngan_01.wav'
)
VUI_VE_GOLDEN_REF = (
    Path(__file__).resolve().parent.parent
    / 'voices'
    / 'vui_ve'
    / 'golden_refs'
    / 'golden_01.wav'
)
VUI_VE_GOLDEN_TEXT = (
    'Và cho đến nay thì tổ chức SCP vẫn chưa tìm được cách để tiêu diệt hoàn '
    'toàn SCP -610 và chỉ có thể dùng robot để tiếp tục quan sát, thu thập data '
    'và nghiên cứu những gì đang diễn ra thôi'
)

# ---------------------------------------------------------------------------
# Model cache (loading the checkpoint costs ~30s and ~2 GB of VRAM)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, object] = {}


def _resolve_alias(name: str) -> str:
    key = name.strip().lower()
    return PRESET_ALIASES.get(key, name.strip())


def _available_device() -> str:
    try:
        import torch

        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def load_vieneu_model(device: str | None = None, backbone: str | None = None):
    """Load (and cache) a VieNeu-TTS v3 Turbo model.

    The ``vieneu`` import and the checkpoint download are deferred until the
    first call so plain ``import core.vieneu_engine`` stays cheap.

    ``backbone`` may be a HF repo id; the Nguyễn Ngọc Ngạn fine-tune is loaded
    with the base repo's tokenizer because its bundled tokenizer is broken.
    """
    from core._vieneu_compat import ensure_neucodec_importable

    ensure_neucodec_importable()

    device = device or _available_device()
    backbone = backbone or BASE_V3_TURBO_REPO
    cache_key = f'{backbone}::{device}'
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from vieneu import Vieneu

    logger.info(
        f'🎙️ [VieNeu] Đang nạp mô hình VieNeu-TTS v3 Turbo ({backbone}, device={device})…'
    )
    if backbone == BASE_V3_TURBO_REPO:
        tts = Vieneu(mode='v3turbo')
    else:
        # Fine-tunes (e.g. Ngạn) ship weights at the repo root and need the base
        # tokenizer; see ``_load_fine_tune``.
        tts = _load_fine_tune(backbone)
    _MODEL_CACHE[cache_key] = tts
    logger.info('🎙️ [VieNeu] Mô hình sẵn sàng.')
    return tts


def _load_fine_tune(backbone_repo: str):
    """Load a v3-Turbo fine-tune whose tokenizer is broken / lives at repo root.

    The upstream ``V3TurboVieNeuTTS`` assumes weights under an ``update/``
    subfolder and reuses the fine-tune's own tokenizer file. The Ngạn fine-tune
    has neither: weights sit at the root and its ``tokenizer.json`` fails to
    instantiate, so we point the low-level engine at the base repo's tokenizer.
    """
    from vieneu._v3_turbo_engine import VieNeuTTSv3Turbo as _LowLevel
    from vieneu.v3turbo import V3TurboVieNeuTTS

    original_init = _LowLevel.__init__

    def patched_init(self, checkpoint_path=BASE_V3_TURBO_REPO, model_subfolder='update',
                     tokenizer_path=None, **kwargs):
        if tokenizer_path is None:
            tokenizer_path = BASE_V3_TURBO_REPO
        # Fine-tune weights live at the repo root, not under ``update/``.
        subfolder = '' if checkpoint_path == backbone_repo else model_subfolder
        return original_init(
            self, checkpoint_path=checkpoint_path, model_subfolder=subfolder,
            tokenizer_path=tokenizer_path, **kwargs,
        )

    _LowLevel.__init__ = patched_init
    try:
        return V3TurboVieNeuTTS(backbone_repo=backbone_repo)
    finally:
        _LowLevel.__init__ = original_init


def list_available_preset_voices(tts=None) -> list[str]:
    """All preset voice names available in the (optionally provided) model."""
    if tts is None:
        try:
            tts = load_vieneu_model()
        except Exception as exc:  # pragma: no cover - offline fallback
            logger.warning(f'⚠️ [VieNeu] Không nạp được mô hình ({exc}); dùng danh sách mặc định.')
            return list(FALLBACK_PRESET_VOICES)
    voices = tts.list_preset_voices()
    # ``list_preset_voices`` returns [(label, name), …] on v3.8.3.
    return [name for _label, name in voices]


@dataclass
class VoiceOption:
    """A user-facing choice in the app's voice dropdown."""

    label: str
    mode: str  # 'preset' | 'classic' | 'clone'
    voice: str | None = None
    ref_audio: Path | None = None
    ref_text: str | None = None


def build_voice_catalogue() -> list[VoiceOption]:
    """Dropdown entries: all presets + Nguyễn Ngọc Ngạn + Vui Vẻ clone."""
    options: list[VoiceOption] = []
    for name in list_available_preset_voices():
        options.append(VoiceOption(label=f'Giọng sẵn có · {name}', mode='preset', voice=name))

    options.append(
        VoiceOption(
            label='Nguyễn Ngọc Ngạn (kinh điển)',
            mode='classic',
            ref_audio=None,  # resolved by the engine at synthesis time
        )
    )
    options.append(
        VoiceOption(
            label='Clone · Vui Vẻ',
            mode='clone',
            ref_audio=VUI_VE_GOLDEN_REF,
            ref_text=VUI_VE_GOLDEN_TEXT,
        )
    )
    options.append(VoiceOption(label='Clone · WAV của bạn', mode='clone'))
    return options


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class VieNeuEngine(TTSEngine):
    """VieNeu-TTS v3 Turbo engine.

    Args:
        mode: ``'preset'``, ``'classic'`` or ``'clone'``.
        voice: preset voice name (aliases like ``'Bình'`` are accepted).
        ref_audio: reference WAV for ``classic`` / ``clone`` modes.
        ref_text: transcript of ``ref_audio`` (optional but improves quality).
        apply_watermark: keep VieNeu's audio watermark. Disabled for QA runs so
            Whisper verification measures the voice, not the watermark.
        device: ``'cuda'`` / ``'cpu'``; auto-detected when omitted.
        batch_size: số chunk gộp vào một forward của VieNeu-TTS v3 Turbo khi
            chạy trên CUDA (CUDA Graph batching). ``None`` → mặc định 4 trên
            ``cuda`` (2–3× nhanh hơn), ``1`` trên CPU.
    """

    def __init__(
        self,
        mode: str = 'preset',
        voice: str | None = None,
        ref_audio: Path | str | None = None,
        ref_text: str | None = None,
        apply_watermark: bool = False,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        if mode not in ('preset', 'classic', 'clone'):
            raise ValueError(f"mode không hợp lệ: {mode!r} (chỉ nhận 'preset', 'classic', 'clone')")
        self.mode = mode
        self.voice = _resolve_alias(voice) if voice else DEFAULT_PRESET_VOICE
        self.ref_audio = Path(ref_audio) if ref_audio else None
        self.ref_text = ref_text
        self.apply_watermark = apply_watermark
        self.device = device
        if batch_size is None:
            resolved_device = device or _available_device()
            batch_size = DEFAULT_BATCH_SIZE_CUDA if resolved_device == 'cuda' else 1
        self.batch_size = max(1, int(batch_size))

    # -- helpers ----------------------------------------------------------
    def _reference_for_classic(self) -> tuple[Path, str | None]:
        """Reference WAV (and optional transcript) for Nguyễn Ngọc Ngạn."""
        if self.ref_audio and Path(self.ref_audio).exists():
            ref_audio = Path(self.ref_audio)
        elif DEFAULT_CLASSIC_NGAN_REF_WAV.exists():
            ref_audio = DEFAULT_CLASSIC_NGAN_REF_WAV
        elif VUI_VE_GOLDEN_REF.exists():
            # Last resort so the mode never hard-fails in a fresh checkout.
            logger.warning(
                '⚠️ [VieNeu] Không tìm thấy WAV mẫu Nguyễn Ngọc Ngạn; '
                'tạm dùng golden ref của Vui Vẻ làm reference clone.'
            )
            return VUI_VE_GOLDEN_REF, VUI_VE_GOLDEN_TEXT
        else:
            raise FileNotFoundError(
                'Không tìm thấy file WAV mẫu cho giọng Nguyễn Ngọc Ngạn. '
                f'Đặt file tại {DEFAULT_CLASSIC_NGAN_REF_WAV} hoặc truyền ref_audio.'
            )

        ref_text = self.ref_text
        if not ref_text:
            sidecar = ref_audio.with_suffix('.txt')
            if sidecar.exists():
                ref_text = sidecar.read_text(encoding='utf-8').strip() or None
        return ref_audio, ref_text

    def describe(self) -> str:
        if self.mode == 'preset':
            return f'preset:{self.voice}'
        if self.mode == 'classic':
            return 'classic:Nguyễn Ngọc Ngạn'
        return f'clone:{self.ref_audio}'

    # -- reference resolution (shared by single & batch synthesis) ---------
    def _resolve_reference(self, tts) -> tuple[object, object | None]:
        """Trả ``(speaker_emb, ref_codes)`` cho giọng hiện tại, một lần cho cả batch."""
        if self.mode == 'preset':
            return tts._resolve_ref(
                voice=self.voice, ref_audio=None, denoise=True, use_ref_codes=True,
            )
        if self.mode == 'classic':
            try:
                if tts is None:  # pragma: no cover - defensive
                    raise RuntimeError('no model')
                return tts._resolve_ref(
                    voice=tts.get_preset_voice(), ref_audio=None,
                    denoise=True, use_ref_codes=True,
                )
            except Exception:  # pragma: no cover - fallback to clone reference
                pass
            ref_audio, _ = self._reference_for_classic()
        else:
            if not self.ref_audio or not Path(self.ref_audio).exists():
                raise FileNotFoundError(
                    f'Chế độ clone cần một file WAV mẫu hợp lệ, nhận được: {self.ref_audio}'
                )
            ref_audio = Path(self.ref_audio)
        return tts._resolve_ref(
            voice=None, ref_audio=str(ref_audio), denoise=True, use_ref_codes=True,
        )

    def synthesize_segments_batch(self, texts: list[str]) -> list[tuple[np.ndarray, int]]:
        """Batch nhiều câu qua một lời gọi ``tts._infer_chunks`` (CUDA graph).

        Trả ``[(waveform_24k_float32, duration_ms), ...]`` đúng thứ tự ``texts``.
        Toàn bộ phoneme của mọi câu được gộp phẳng rồi chia nhóm ``batch_size``,
        nên các câu ngắn vẫn lấp đầy một forward (thay vì 1 chunk/lần như
        ``synthesize_segment``).
        """
        if not texts:
            return []

        tts = load_vieneu_model(self.device)
        speaker_emb, ref_codes = self._resolve_reference(tts)
        orig_sr = int(getattr(tts, 'sample_rate', 48000))

        all_phonemes: list[str] = []
        per_text: list[tuple[int, int, list[float]]] = []
        for text in texts:
            chunks, gaps = _upstream_chunker()(text, max_chars=256)
            start = len(all_phonemes)
            all_phonemes.extend(_upstream_phonemize()(c) for c in chunks)
            per_text.append((start, len(all_phonemes), list(gaps)))

        if not all_phonemes:
            return [(_write_silence_array(), 200) for _ in texts]

        wavs = tts._infer_chunks(
            all_phonemes,
            speaker_emb,
            ref_codes,
            True,
            batch_size=self.batch_size,
            sampling=dict(DEFAULT_SAMPLING),
        )

        join_audio_chunks, gaps_to_silence = _upstream_join()
        results: list[tuple[np.ndarray, int]] = []
        for start, end, gaps in per_text:
            if start == end:
                # Câu rỗng/toàn khoảng trắng không sinh chunk nào → im lặng 200 ms.
                results.append((_write_silence_array(), 200))
                continue
            wav = join_audio_chunks(
                wavs[start:end], orig_sr, silence_ps=gaps_to_silence(gaps)
            )
            wav = _resample_waveform_in_memory(wav, orig_sr, SAMPLE_RATE)
            duration_ms = int(round(len(wav) / float(SAMPLE_RATE) * 1000))
            results.append((wav, duration_ms))
        return results

    # -- synthesis --------------------------------------------------------
    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        if not text or not text.strip():
            return _write_silence(out_wav, milliseconds=200)

        import soundfile as sf

        if self.mode == 'classic':
            # Nguyễn Ngọc Ngạn: prefer the real fine-tune backbone; if it can't
            # be loaded (offline / gated), fall back to an instant clone from a
            # reference WAV so the mode never hard-fails.
            try:
                tts = load_vieneu_model(self.device, backbone=NGAN_FINE_TUNE_REPO)
                wav = tts.infer(
                    text, voice=tts.get_preset_voice(),
                    apply_watermark=self.apply_watermark,
                    batch_size=self.batch_size,
                )
                sample_rate = int(getattr(tts, 'sample_rate', 48000))
                return _write_wav_resampled(wav, sample_rate, out_wav)
            except Exception as exc:  # pragma: no cover - offline fallback
                logger.warning(
                    f'⚠️ [VieNeu] Không nạp được fine-tune Ngạn ({exc}); '
                    'chuyển sang instant clone từ WAV mẫu.'
                )

        tts = load_vieneu_model(self.device)
        infer_kwargs: dict = {
            'apply_watermark': self.apply_watermark,
            'batch_size': self.batch_size,
        }

        if self.mode == 'preset':
            infer_kwargs['voice'] = self.voice
            wav = tts.infer(text, **infer_kwargs)
        else:
            if self.mode == 'classic':
                ref_audio, ref_text = self._reference_for_classic()
            else:
                if not self.ref_audio or not Path(self.ref_audio).exists():
                    raise FileNotFoundError(
                        f'Chế độ clone cần một file WAV mẫu hợp lệ, nhận được: {self.ref_audio}'
                    )
                ref_audio = Path(self.ref_audio)
                ref_text = self.ref_text
            infer_kwargs['ref_audio'] = str(ref_audio)
            if ref_text:
                infer_kwargs['ref_text'] = ref_text
            wav = tts.infer(text, **infer_kwargs)

        sample_rate = int(getattr(tts, 'sample_rate', 48000))
        return _write_wav_resampled(wav, sample_rate, out_wav)


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------
def _write_silence(out_wav: Path, milliseconds: int) -> int:
    import wave

    with wave.open(str(out_wav), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        frames = int(SAMPLE_RATE * milliseconds / 1000.0)
        wf.writeframes(b'\x00\x00' * frames)
    return milliseconds


def _resample_waveform_in_memory(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample mono waveform trong RAM (soxr → torchaudio → scipy).

    Không ghi đĩa, không subprocess ở đường thường. Trả 1-D ``float32``.
    """
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    samples = samples.astype(np.float32, copy=False)
    if orig_sr == target_sr or samples.size == 0:
        return samples

    try:
        import soxr

        return np.asarray(soxr.resample(samples, orig_sr, target_sr), dtype=np.float32)
    except Exception as exc:  # pragma: no cover - soxr hiếm khi lỗi
        logger.debug(f'[VieNeu] soxr resample lỗi ({exc}); thử torchaudio.')

    try:  # pragma: no cover - torchaudio thường không được chạy tới
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(samples)
        out = F.resample(tensor, orig_freq=orig_sr, new_freq=target_sr)
        return out.numpy().astype(np.float32, copy=False)
    except Exception as exc:  # pragma: no cover
        logger.debug(f'[VieNeu] torchaudio resample lỗi ({exc}); thử scipy.')

    try:  # pragma: no cover
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(orig_sr), int(target_sr))
        return resample_poly(samples, target_sr // g, orig_sr // g).astype(np.float32, copy=False)
    except Exception as exc:  # pragma: no cover
        logger.warning(f'⚠️ [VieNeu] In-memory resample thất bại ({exc}); dùng ffmpeg.')
        raise


def _resample_via_ffmpeg(samples: np.ndarray, sample_rate: int, out_wav: Path) -> int:
    """Fallback cuối: ghi WAV tạm rồi gọi FFmpeg (hành vi cũ)."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpd:
        raw = Path(tmpd) / 'raw.wav'
        sf.write(str(raw), samples.astype(np.float32), sample_rate)
        ffmpeg_bin = _get_ffmpeg_bin()
        subprocess.run(
            [
                ffmpeg_bin, '-y', '-v', 'error',
                '-i', str(raw),
                '-ac', '1', '-ar', str(SAMPLE_RATE), '-sample_fmt', 's16',
                str(out_wav),
            ],
            check=True,
        )
    return _wav_duration_ms(out_wav)


def _concat_waveforms(wavs: list[np.ndarray]) -> np.ndarray:
    """Nối các waveform 1-D float32 của một câu (nhiều chunk) thành một mảng."""
    arrays = [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs if w is not None]
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays)


def _write_silence_array(milliseconds: int = 200) -> np.ndarray:
    """Mảng im lặng ``float32`` tại ``SAMPLE_RATE`` (dùng cho câu rỗng)."""
    return np.zeros(int(SAMPLE_RATE * milliseconds / 1000.0), dtype=np.float32)


def _write_wav_resampled(samples: np.ndarray, sample_rate: int, out_wav: Path) -> int:
    """Write ``samples`` as mono 16-bit PCM at ``SAMPLE_RATE`` (in-memory resample)."""
    import soundfile as sf

    samples = np.asarray(samples)
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    if samples.size == 0:
        return _write_silence(out_wav, milliseconds=200)

    try:
        resampled = _resample_waveform_in_memory(samples, sample_rate, SAMPLE_RATE)
    except Exception:  # pragma: no cover - chỉ khi mọi backend trong RAM hỏng
        return _resample_via_ffmpeg(samples, sample_rate, out_wav)

    sf.write(str(out_wav), resampled, SAMPLE_RATE, subtype='PCM_16')
    return _wav_duration_ms(out_wav)


def synthesize_text_to_wav(
    text: str,
    out_wav: Path,
    engine: VieNeuEngine | None = None,
) -> int:
    """Convenience helper: synthesize one string with a VieNeu engine."""
    engine = engine or VieNeuEngine()
    return engine.synthesize_segment(text, out_wav)


__all__ = [
    'DEFAULT_BATCH_SIZE_CUDA',
    'DEFAULT_CLASSIC_NGAN_REF_WAV',
    'DEFAULT_PRESET_VOICE',
    'PRESET_ALIASES',
    'VUI_VE_GOLDEN_REF',
    'VUI_VE_GOLDEN_TEXT',
    'VieNeuEngine',
    'VoiceOption',
    'build_voice_catalogue',
    'list_available_preset_voices',
    'load_vieneu_model',
    'synthesize_text_to_wav',
]
