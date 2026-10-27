import shutil
import subprocess
import tempfile
from pathlib import Path
from core.models import BookMetadata
from core.tts_engine import ChapterAudioResult


def _get_ffmpeg_bin() -> str:
    venv_ffmpeg = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'ffmpeg'
    if venv_ffmpeg.exists():
        return str(venv_ffmpeg)
    return shutil.which('ffmpeg') or 'ffmpeg'


def _escape_ffmeta(val: str) -> str:
    escaped = val.replace('\\', '\\\\')
    for ch in ('=', ';', '#'):
        escaped = escaped.replace(ch, f'\\{ch}')
    return escaped.replace('\n', ' ')


def build_ffmetadata(meta: BookMetadata, chapters: list[ChapterAudioResult]) -> str:
    lines = [
        ';FFMETADATA1',
        f'title={_escape_ffmeta(meta.title)}',
        f'artist={_escape_ffmeta(meta.author)}',
        '',
    ]
    offset_ms = 0
    for ch in chapters:
        start_ms = offset_ms
        end_ms = offset_ms + max(1, ch.duration_ms)
        lines.extend(
            [
                '[CHAPTER]',
                'TIMEBASE=1/1000',
                f'START={start_ms}',
                f'END={end_ms}',
                f'title={_escape_ffmeta(ch.title)}',
                '',
            ]
        )
        offset_ms = end_ms
    return '\n'.join(lines)


def _format_vtt_time(ms: int) -> str:
    total_sec, millis = divmod(max(0, int(ms)), 1000)
    hours, rem = divmod(total_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}'


def export_webvtt(chapters: list[ChapterAudioResult], out_vtt: Path) -> Path:
    out_vtt = Path(out_vtt)
    out_vtt.parent.mkdir(parents=True, exist_ok=True)
    lines = ['WEBVTT', '']
    offset_ms = 0
    cue_idx = 1
    for ch in chapters:
        for cue in ch.cues:
            start_str = _format_vtt_time(offset_ms + cue.start_ms)
            end_str = _format_vtt_time(offset_ms + cue.end_ms)
            lines.append(str(cue_idx))
            lines.append(f'{start_str} --> {end_str}')
            lines.append(cue.text)
            lines.append('')
            cue_idx += 1
        offset_ms += ch.duration_ms
    out_vtt.write_text('\n'.join(lines), encoding='utf-8')
    return out_vtt


def package_m4b(meta: BookMetadata, chapters: list[ChapterAudioResult], out_m4b: Path) -> Path:
    if not chapters:
        raise ValueError('Cần ít nhất 1 chương để đóng gói M4B.')
    out_m4b = Path(out_m4b)
    out_m4b.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _get_ffmpeg_bin()

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_dir = Path(tmpd)
        concat_file = tmp_dir / 'concat.txt'
        concat_lines = []
        for ch in chapters:
            safe_path = str(Path(ch.audio_path).resolve()).replace("'", "'\\''")
            concat_lines.append(f"file '{safe_path}'")
        concat_file.write_text('\n'.join(concat_lines) + '\n', encoding='utf-8')

        ffmeta_file = tmp_dir / 'ffmetadata.txt'
        ffmeta_file.write_text(build_ffmetadata(meta, chapters), encoding='utf-8')

        has_cover = meta.cover_path is not None and Path(meta.cover_path).exists()
        cmd = [
            ffmpeg_bin, '-y', '-v', 'error',
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-f', 'ffmetadata', '-i', str(ffmeta_file),
        ]
        if has_cover:
            cmd.extend(['-i', str(Path(meta.cover_path).resolve())])
            cmd.extend([
                '-map', '0:a',
                '-map', '2:v',
                '-map_metadata', '1',
                '-map_chapters', '1',
                '-c:a', 'aac', '-b:a', '64k', '-ac', '1',
                '-c:v', 'copy',
                '-disposition:v:0', 'attached_pic',
            ])
        else:
            cmd.extend([
                '-map', '0:a',
                '-map_metadata', '1',
                '-map_chapters', '1',
                '-c:a', 'aac', '-b:a', '64k', '-ac', '1',
            ])
        cmd.extend(['-movflags', '+faststart', str(out_m4b)])
        subprocess.run(cmd, check=True)

    return out_m4b
