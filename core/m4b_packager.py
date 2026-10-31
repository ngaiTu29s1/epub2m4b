import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from core.models import BookMetadata
from core.tts_engine import ChapterAudioResult

logger = logging.getLogger('epub2m4b.packager')


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
    total_cues = sum(len(ch.cues) for ch in chapters)
    out_vtt.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f'📝 [VTT] Xuất file phụ đề WebVTT ({total_cues} cues) -> {out_vtt.name}')
    return out_vtt


def encode_wav_to_m4a(
    wav_path: Path,
    out_m4a: Path | None = None,
    force: bool = False,
    bitrate: str = '64k',
) -> Path:
    """Nén một file WAV thành M4A/AAC (tách khỏi bước đóng gói).

    Tách riêng để chạy song song với vòng render GPU. Kết quả được cache: file
    đích còn mới hơn WAV và có dữ liệu thì bỏ qua encode.
    """
    wav_path = Path(wav_path)
    out_m4a = Path(out_m4a) if out_m4a else wav_path.with_suffix('.m4a')

    if (
        not force
        and out_m4a.exists()
        and out_m4a.stat().st_size > 0
        and out_m4a.stat().st_mtime >= wav_path.stat().st_mtime
    ):
        logger.info(f'⚡ [AAC CACHE] Đã có {out_m4a.name}, bỏ qua encode.')
        return out_m4a

    out_m4a.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _get_ffmpeg_bin(), '-y', '-v', 'error',
        '-i', str(wav_path),
        '-c:a', 'aac', '-b:a', bitrate, '-ac', '1',
        '-movflags', '+faststart',
        str(out_m4a),
    ]
    logger.info(f'🎧 [AAC] Encode {wav_path.name} → {out_m4a.name} ({bitrate})...')
    subprocess.run(cmd, check=True)
    return out_m4a


def _m4a_for(ch: ChapterAudioResult) -> Path | None:
    """File .m4a đã encode sẵn cho chương, nếu có và hợp lệ (size > 0)."""
    audio = Path(ch.audio_path)
    candidate = audio if audio.suffix.lower() == '.m4a' else audio.with_suffix('.m4a')
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    return None


def resolve_package_sources(
    chapters: list[ChapterAudioResult],
) -> tuple[list[Path], bool]:
    """Chọn nguồn cho bước đóng gói.

    Trả ``(danh_sách_file, dùng_stream_copy)``. Stream copy chỉ khi **mọi**
    chương đều có .m4a sẵn — trộn WAV và M4A trong một lệnh concat sẽ vỡ.
    """
    sources: list[Path] = []
    for ch in chapters:
        m4a = _m4a_for(ch)
        if m4a is None:
            return [Path(c.audio_path) for c in chapters], False
        sources.append(m4a)
    return sources, True


def package_m4b(meta: BookMetadata, chapters: list[ChapterAudioResult], out_m4b: Path) -> Path:
    if not chapters:
        raise ValueError('Cần ít nhất 1 chương để đóng gói M4B.')
    out_m4b = Path(out_m4b)
    out_m4b.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = _get_ffmpeg_bin()
    logger.info(f'🎬 [FFMPEG] Bắt đầu ghép {len(chapters)} chương vào file {out_m4b.name}...')

    sources, stream_copy = resolve_package_sources(chapters)
    if stream_copy:
        logger.info(
            f'⚡ [PIPELINE] Phát hiện đủ {len(sources)} file .m4a — dùng stream copy '
            f'(-c:a copy), bỏ qua re-encode AAC.'
        )
    else:
        logger.info('🎙️ [PIPELINE] Thiếu .m4a cho một số chương — fallback về WAV + encode AAC.')

    # Nhãn chương theo thời lượng WAV gốc (không đổi giữa 2 chế độ).
    concat_sources = sources

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_dir = Path(tmpd)
        concat_file = tmp_dir / 'concat.txt'
        concat_lines = []
        for src in concat_sources:
            safe_path = str(Path(src).resolve()).replace("'", "'\\''")
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
        audio_args = (
            ['-c:a', 'copy']
            if stream_copy
            else ['-c:a', 'aac', '-b:a', '64k', '-ac', '1']
        )
        if has_cover:
            cmd.extend(['-i', str(Path(meta.cover_path).resolve())])
            cmd.extend([
                '-map', '0:a',
                '-map', '2:v',
                '-map_metadata', '1',
                '-map_chapters', '1',
                *audio_args,
                '-c:v', 'copy',
                '-disposition:v:0', 'attached_pic',
            ])
        else:
            cmd.extend([
                '-map', '0:a',
                '-map_metadata', '1',
                '-map_chapters', '1',
                *audio_args,
            ])
        cmd.extend(['-movflags', '+faststart', str(out_m4b)])
        subprocess.run(cmd, check=True)

    logger.info(f'✨ [FFMPEG] Đóng gói thành công file sách nói: {out_m4b}')
    return out_m4b
