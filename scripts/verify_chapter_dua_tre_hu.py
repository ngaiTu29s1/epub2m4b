"""Verify fine-tuned F5-TTS model on a real chapter from 'Đứa Trẻ Hư'."""
from pathlib import Path
import json
import logging
from core.epub_parser import parse_book_markdown
from core.tts_engine import F5TTSEngine, render_chapter
from faster_whisper import WhisperModel
import soundfile as sf
import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_chapter")

def verify_on_real_chapter(ckpt_path: Path | None = None) -> Path:
    book_md = Path("workspace/dua_tre_hu_tu_kim_tran/book.md")
    if not book_md.exists():
        raise FileNotFoundError(f"Không tìm thấy file sách: {book_md}")

    # Parse chapters from markdown
    chapters = parse_book_markdown(book_md)
    # Pick 'PHẦN DẪN' or chapter 1
    test_chapter = next((c for c in chapters if "DẪN" in c.title.upper()), chapters[0])
    logger.info(f"📖 Chọn chương test: '{test_chapter.title}' ({len(test_chapter.markdown)} ký tự)")

    # Read profile for golden ref
    profile_path = Path("voices/vui_ve/profile.json")
    with open(profile_path, encoding="utf-8") as f:
        prof = json.load(f)

    ref_wav = Path(prof.get("best_golden_ref_wav") or "voices/vui_ve/golden_refs/golden_01.wav")
    if not ref_wav.is_absolute():
        ref_wav = Path("voices/vui_ve") / ref_wav
    ref_text = prof.get("best_golden_ref_text", "")

    # Initialize F5-TTS engine with fine-tuned checkpoint
    engine = F5TTSEngine(ref_audio=ref_wav, ref_text=ref_text, ckpt_path=ckpt_path)

    out_dir = Path("voices/vui_ve/test_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🎙️ Bắt đầu render toàn bộ chương bằng F5-TTS...")
    res = render_chapter(test_chapter, engine, out_dir, force=True)
    logger.info(f"✔️ Render hoàn tất: {res.audio_path} ({res.duration_ms / 1000:.1f}s, {len(res.cues)} cues)")

    # Transcribe back with Whisper to verify Vietnamese accuracy
    logger.info("🔍 Chạy Faster-Whisper kiểm chứng độ chính xác tiếng Việt...")
    audio, sr = sf.read(str(res.audio_path))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    asr = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = asr.transcribe(audio, language="vi")
    recognized = " ".join([s.text for s in segs]).strip()
    logger.info(f"📝 Whisper ASR nghe được: {recognized[:300]}...")

    report_file = Path("voices/vui_ve/CHAPTER_VERIFICATION.md")
    report_file.write_text(
        f"# Báo cáo Kiểm thử Thực tế trên Sách 'Đứa Trẻ Hư'\n\n"
        f"- **Chương:** {test_chapter.title}\n"
        f"- **Thời lượng audio:** {res.duration_ms / 1000:.1f} giây\n"
        f"- **Số phân đoạn (cues):** {len(res.cues)}\n"
        f"- **Đường dẫn file audio:** `{res.audio_path}`\n\n"
        f"## Nội dung Whisper ASR nhận diện được:\n> {recognized}\n",
        encoding="utf-8",
    )
    logger.info(f"📄 Đã lưu báo cáo kiểm thử tại {report_file}")
    return res.audio_path

if __name__ == "__main__":
    verify_on_real_chapter()
