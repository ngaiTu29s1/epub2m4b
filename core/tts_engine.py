import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
import edge_tts
from core.models import Chapter
from core.text_normalizer import markdown_to_segments

logger = logging.getLogger('epub2m4b.tts')

SAMPLE_RATE = 24000


@dataclass
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class ChapterAudioResult:
    chapter_index: int
    title: str
    audio_path: Path
    duration_ms: int
    cues: list[SubtitleCue]


class TTSEngine:
    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        raise NotImplementedError


def _get_ffmpeg_bin() -> str:
    venv_ffmpeg = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'ffmpeg'
    if venv_ffmpeg.exists():
        return str(venv_ffmpeg)
    return shutil.which('ffmpeg') or 'ffmpeg'


def _wav_duration_ms(wav_path: Path) -> int:
    with wave.open(str(wav_path), 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return int(round((frames / float(rate)) * 1000))


class EdgeTTSEngine(TTSEngine):
    def __init__(self, voice: str = 'vi-VN-NamMinhNeural', rate: str = '+0%'):
        self.voice = voice
        self.rate = rate

    async def _save_mp3(self, text: str, mp3_path: Path, max_retries: int = 5) -> None:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                comm = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
                await comm.save(str(mp3_path))
                return
            except edge_tts.exceptions.NoAudioReceived as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    break
                delay = 0.5 * (2 ** attempt)
                logger.warning(
                    f'⚠️ [RETRY] Edge-TTS gặp lỗi NoAudioReceived (thử lại {attempt + 1}/{max_retries}) '
                    f'sau {delay:.1f}s...'
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    break
                delay = 0.5 * (2 ** attempt)
                logger.warning(
                    f'⚠️ [RETRY] Edge-TTS gặp ngoại lệ ({type(exc).__name__}: {exc}) '
                    f'thử lại {attempt + 1}/{max_retries} sau {delay:.1f}s...'
                )
                await asyncio.sleep(delay)
        if last_exc:
            raise last_exc

    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if not re.search(r'\w', text):
            with wave.open(str(out_wav), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b'\x00\x00' * int(SAMPLE_RATE * 0.2))
            return 200

        with tempfile.TemporaryDirectory() as tmpd:
            mp3_path = Path(tmpd) / 'seg.mp3'
            asyncio.run(self._save_mp3(text, mp3_path))
            ffmpeg_bin = _get_ffmpeg_bin()
            subprocess.run(
                [
                    ffmpeg_bin, '-y', '-v', 'error',
                    '-i', str(mp3_path),
                    '-ac', '1', '-ar', str(SAMPLE_RATE), '-sample_fmt', 's16',
                    str(out_wav),
                ],
                check=True,
            )

        return _wav_duration_ms(out_wav)


def render_chapter(
    chapter: Chapter,
    engine: TTSEngine,
    out_dir: Path,
    force: bool = False,
) -> ChapterAudioResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f'chapter_{chapter.index:03d}.wav'
    meta_path = out_dir / f'chapter_{chapter.index:03d}.json'

    if not force and wav_path.exists() and meta_path.exists():
        data = json.loads(meta_path.read_text(encoding='utf-8'))
        cues = [SubtitleCue(**c) for c in data['cues']]
        logger.info(
            f'⚡ [CACHE] Chương {chapter.index:03d} "{chapter.title}" đã có cache audio '
            f'({data["duration_ms"] / 1000:.1f}s), bỏ qua render.'
        )
        return ChapterAudioResult(
            chapter_index=chapter.index,
            title=data['title'],
            audio_path=wav_path,
            duration_ms=data['duration_ms'],
            cues=cues,
        )

    segments = markdown_to_segments(chapter.markdown)
    cues: list[SubtitleCue] = []
    current_ms = 0
    logger.info(
        f'🎙️ [TTS] Chương {chapter.index:03d} "{chapter.title}": Bắt đầu render {len(segments)} phân đoạn...'
    )

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_dir = Path(tmpd)
        with wave.open(str(wav_path), 'wb') as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)
            out_wf.setframerate(SAMPLE_RATE)

            for i, seg in enumerate(segments):
                seg_wav = tmp_dir / f'seg_{i:04d}.wav'
                seg_dur_ms = engine.synthesize_segment(seg.text, seg_wav)
                with wave.open(str(seg_wav), 'rb') as in_wf:
                    out_wf.writeframes(in_wf.readframes(in_wf.getnframes()))

                cues.append(
                    SubtitleCue(
                        start_ms=current_ms,
                        end_ms=current_ms + seg_dur_ms,
                        text=seg.text,
                    )
                )
                current_ms += seg_dur_ms

                if seg.pause_after_ms > 0:
                    silence_frames = int(SAMPLE_RATE * (seg.pause_after_ms / 1000.0))
                    out_wf.writeframes(b'\x00\x00' * silence_frames)
                    current_ms += seg.pause_after_ms

    payload = {
        'chapter_index': chapter.index,
        'title': chapter.title,
        'duration_ms': current_ms,
        'cues': [asdict(c) for c in cues],
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(
        f'✔️ [TTS] Chương {chapter.index:03d} "{chapter.title}" hoàn thành '
        f'({current_ms / 1000:.1f}s, {len(cues)} cues).'
    )

    return ChapterAudioResult(
        chapter_index=chapter.index,
        title=chapter.title,
        audio_path=wav_path,
        duration_ms=current_ms,
        cues=cues,
    )


# --------------------------------------------------------------------------
# Lazy re-export of the VieNeu-TTS engine.
# --------------------------------------------------------------------------
# ``core.vieneu_engine`` imports this module (for ``TTSEngine`` and the WAV
# helpers), so importing it at module top-level would create a cycle. Exposing
# the engine through ``__getattr__`` lets callers write
# ``from core.tts_engine import VieNeuEngine`` while keeping the heavy
# vieneu/torch imports lazy until the engine is actually used.
_LAZY_ENGINE_EXPORTS = {
    'VieNeuEngine': 'core.vieneu_engine',
}

__all__ = [
    'ChapterAudioResult',
    'EdgeTTSEngine',
    'SAMPLE_RATE',
    'SubtitleCue',
    'TTSEngine',
    'VieNeuEngine',
    'render_chapter',
]


def __getattr__(name: str):
    module_path = _LAZY_ENGINE_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    import importlib

    return getattr(importlib.import_module(module_path), name)
