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
    def __init__(
        self,
        voice: str = 'vi-VN-NamMinhNeural',
        rate: str = '+0%',
        rvc_model_path: Path | None = None,
        rvc_index_path: Path | None = None,
    ):
        self.voice = voice
        self.rate = rate
        self.rvc_model_path = Path(rvc_model_path) if rvc_model_path else None
        self.rvc_index_path = Path(rvc_index_path) if rvc_index_path else None

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

        if self.rvc_model_path and self.rvc_model_path.exists():
            self._apply_rvc(out_wav)

        return _wav_duration_ms(out_wav)

    def _apply_rvc(self, wav_path: Path) -> None:
        try:
            from rvc_python.infer import RVCInference
        except ImportError as exc:
            raise RuntimeError(
                'Cần cài thêm rvc-python trong venv để chạy chế độ Edge-TTS + RVC.'
            ) from exc
        rvc = RVCInference(device='cuda:0')
        rvc.load_model(str(self.rvc_model_path), index_path=str(self.rvc_index_path or ''))
        tmp_out = wav_path.with_suffix('.rvc.wav')
        rvc.infer_file(str(wav_path), str(tmp_out))
        ffmpeg_bin = _get_ffmpeg_bin()
        subprocess.run(
            [
                ffmpeg_bin, '-y', '-v', 'error',
                '-i', str(tmp_out),
                '-ac', '1', '-ar', str(SAMPLE_RATE), '-sample_fmt', 's16',
                str(wav_path),
            ],
            check=True,
        )
        tmp_out.unlink(missing_ok=True)


_F5_CACHE: dict = {}


def _get_f5_resources(ckpt_path: Path | None = None):
    key = str(ckpt_path) if ckpt_path else 'default'
    if key in _F5_CACHE:
        return _F5_CACHE[key]

    import soundfile as sf
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download

    def sf_load(uri, *args, **kwargs):
        data, sr = sf.read(str(uri), dtype='float32')
        t = torch.from_numpy(data)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        else:
            t = t.t()
        return t, sr

    def sf_save(uri, src, sample_rate, *args, **kwargs):
        arr = src.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T
        sf.write(str(uri), arr, sample_rate)

    torchaudio.load = sf_load
    torchaudio.save = sf_save

    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    vocoder = load_vocoder(is_local=False)
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    if ckpt_path and Path(ckpt_path).exists():
        real_ckpt = str(ckpt_path)
    else:
        real_ckpt = hf_hub_download(
            repo_id='SWivid/F5-TTS', filename='F5TTS_v1_Base/model_1250000.safetensors'
        )

    model = load_model(model_cls, model_cfg, ckpt_path=real_ckpt, device=device)
    _F5_CACHE[key] = (model, vocoder, device)
    return _F5_CACHE[key]


def _get_f5_cli() -> str | None:
    venv_cli = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'f5-tts_infer-cli'
    if venv_cli.exists():
        return str(venv_cli)
    return shutil.which('f5-tts_infer-cli')


class F5TTSEngine(TTSEngine):
    def __init__(self, ref_audio: Path, ref_text: str, ckpt_path: Path | None = None):
        self.ref_audio = Path(ref_audio)
        self.ref_text = ref_text
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None

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

        try:
            import soundfile as sf
            from f5_tts.infer.utils_infer import infer_process

            model, vocoder, device = _get_f5_resources(self.ckpt_path)
            wav, sr, _ = infer_process(
                ref_audio=str(self.ref_audio),
                ref_text=self.ref_text,
                gen_text=text,
                model_obj=model,
                vocoder=vocoder,
                mel_spec_type='vocos',
                device=device,
            )
            with tempfile.TemporaryDirectory() as tmpd:
                raw_wav = Path(tmpd) / 'raw.wav'
                sf.write(str(raw_wav), wav, sr)
                ffmpeg_bin = _get_ffmpeg_bin()
                subprocess.run(
                    [
                        ffmpeg_bin, '-y', '-v', 'error',
                        '-i', str(raw_wav),
                        '-ac', '1', '-ar', str(SAMPLE_RATE), '-sample_fmt', 's16',
                        str(out_wav),
                    ],
                    check=True,
                )
            return _wav_duration_ms(out_wav)
        except Exception as exc:
            logger.warning(f'⚠️ [F5-TTS] In-process inference gặp lỗi ({exc}), chuyển sang CLI fallback...')
            cli = _get_f5_cli()
            if not cli:
                raise RuntimeError(
                    f'F5-TTS inference thất bại: {exc}. Hãy kiểm tra cấu hình F5-TTS.'
                ) from exc

            with tempfile.TemporaryDirectory() as tmpd:
                cmd = [
                    cli,
                    '--ref_audio', str(self.ref_audio),
                    '--ref_text', self.ref_text,
                    '--gen_text', text,
                    '--output_dir', tmpd,
                ]
                if self.ckpt_path and self.ckpt_path.exists():
                    cmd.extend(['--ckpt_file', str(self.ckpt_path)])
                subprocess.run(cmd, check=True)
                generated = list(Path(tmpd).glob('*.wav'))
                if not generated:
                    raise RuntimeError('F5-TTS không sinh ra file wav.')
                ffmpeg_bin = _get_ffmpeg_bin()
                subprocess.run(
                    [
                        ffmpeg_bin, '-y', '-v', 'error',
                        '-i', str(generated[0]),
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
