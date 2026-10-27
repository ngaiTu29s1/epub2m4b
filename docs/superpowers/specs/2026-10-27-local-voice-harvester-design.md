# Thiết kế Chi tiết: Autonomous Local Voice Harvester (`core/voice_harvester.py`)

- **Ngày:** 2026-10-27
- **Phạm vi:** Sub-project 1 — Tự động tìm kiếm, tải âm thanh YouTube, tách nhạc nền, lọc câu chuẩn phòng thu ("4 Chốt chặn Emilia-Pipe") và xuất gói dữ liệu sẵn sàng cho `F5-TTS` (Zero-shot & Fine-tune) / `GPT-SoVITS` / `RVC v2` ngay trên máy Local (**RTX 3060 12GB VRAM**).

---

## 1. Mục tiêu & Quy trình 5 Bước ("Gạn đục khơi trong")

Người dùng chỉ cần cung cấp `voice_id` (ví dụ: `"vui_ve"`), `query_or_url` (từ khóa tìm kiếm `"kênh Vui Vẻ"`, URL Channel/Playlist, hoặc danh sách URL video), và `target_clean_minutes` (mặc định `30.0` phút).

Hệ thống tự động chạy 5 bước trên máy Local:

1. **Bước 1 – Smart YouTube Discovery & Dedup (`discover_and_download_videos`):**
   - Dùng `yt-dlp` tìm các video mới nhất cùng thời kỳ (để đồng nhất màu micro).
   - Loại bỏ video quá ngắn (`< 120s`, tránh Shorts) hoặc quá dài (`> 3600s`, tránh Livestream tạp âm), và bỏ qua các `video_id` đã nằm trong `voices/<voice_id>/profile.json`.
   - Cắt bỏ `15s` đầu (Intro) và `15s` cuối (Outro) khi trích xuất WAV `24000 Hz` Mono.

2. **Bước 2 – Vocal Isolation & Chốt chặn 1: `Vocal-to-BGM Ratio >= 8.0 dB` (`separate_and_score_vocals`):**
   - Dùng `audio-separator` (UVR5 MDX-Net trên RTX 3060, hoặc fallback bộ lọc phổ/FFmpeg khi chạy test) tách file gốc thành `vocal.wav` và `inst.wav`.
   - Tính năng lượng RMS (dB) trên từng khung thời gian: bất kỳ đoạn nào có nhạc nền quá to (`rms_db(vocal) - rms_db(inst) < 8.0 dB`) sẽ bị đánh dấu loại bỏ để tránh hiện tượng "bóng ma khuyết tần số" của UVR5.

3. **Bước 3 – Smart VAD Slicing & Chốt chặn 2: `+200ms Padding` (`slice_Clean_segments_with_padding`):**
   - Phát hiện khoảng lặng để cắt thành các câu hoàn chỉnh dài từ `3.5s` đến `12.0s`.
   - Luôn giữ lại **`+200ms` đệm đầu và `+200ms` đệm cuối** (`pad_ms=200`) để bảo toàn tuyệt đối các phụ âm cuối Tiếng Việt (`-c, -t, -p, -ng`).
   - Loại bỏ câu bị vỡ tiếng (`peak >= -0.2 dBFS`) hoặc quá nhỏ (`rms_db < -42.0 dBFS`).

4. **Bước 4 – Dominant Speaker Clustering (`filter_dominant_speaker`):**
   - Trích xuất đặc trưng âm sắc (Speaker Embedding / MFCC-Pitch-Spectral centroid) cho tất cả các câu vừa cắt.
   - Tìm tâm cụm của người nói chính (**Dominant Speaker Centroid**) và tự động loại bỏ mọi câu có độ tương đồng `cosine_similarity < 0.78` (loại bỏ giọng khách mời, phỏng vấn, quảng cáo chèn giữa video).

5. **Bước 5 – Whisper ASR & Chốt chặn 3: `Word Confidence >= 0.92` + Multi-Format Export (`transcribe_and_export_voice_pack`):**
   - Chạy `faster-whisper` (`large-v3` hoặc `large-v3-turbo` trên RTX 3060 FP16, `word_timestamps=True`).
   - Loại bỏ ngay lập tức bất kỳ câu nào có `min(word.probability) < 0.92` hoặc tốc độ nói bất thường (`< 8` hoặc `> 28` ký tự/giây).
   - **Tự bầu chọn Top 3 "Golden 10s Prompts" (`golden_refs/`):** Chọn 3 câu dài `7.0s – 11.0s` có điểm tổng hợp cao nhất (SNR cao nhất + `word_confidence` cao nhất + chứa tối thiểu 4/6 dấu thanh Tiếng Việt) để dùng ngay cho **Zero-shot `F5-TTS`**.
   - **Xuất chuẩn Fine-tune Dataset (`voices/<voice_id>/`):**
     - `wavs/clip_XXXX.wav` (`24000 Hz`, 16-bit PCM, Mono)
     - `metadata.csv` (`audio_file|text` chuẩn `F5-TTS`)
     - `gpt_sovits.list` (`wav_path|speaker|VI|text` chuẩn `GPT-SoVITS`)
     - `profile.json` (lưu lịch sử `processed_video_ids`, tổng thời lượng sạch, đường dẫn `best_golden_ref_wav` và `best_golden_ref_text`).

---

## 2. Ước tính Phần cứng & Thời gian Thực tế trên Máy Local của Bạn

Với cấu hình máy hiện tại (**GPU NVIDIA RTX 3060 12GB VRAM + 16GB RAM + WSL2**), để gom được **30 phút Dataset "Vàng"** (cần tải và lọc từ khoảng **75 phút video gốc** trên YouTube, tức khoảng 5–6 video dài 12–15 phút của kênh *Vui Vẻ*):

| Công đoạn | Thiết bị xử lý chính | Mức ăn Tài nguyên (VRAM / Disk) | Thời gian thực tế (cho 75 phút video gốc $\rightarrow$ 30 phút sạch) |
| :--- | :--- | :--- | :---: |
| **1. Search & Tải 75 phút Audio (`yt-dlp`)** | Network + CPU | ~0 GB VRAM / ~800 MB Disk tạm | **~1.5 – 3 phút** |
| **2. Tách nhạc nền UVR5 (`MDX-Net` FP16)** | **RTX 3060 GPU** | **~3.2 GB VRAM** | **~3 – 4 phút** *(nhanh gấp ~20x thời gian thực)* |
| **3. Cắt câu VAD (`+200ms pad`) + Đo SNR `>=8dB`** | CPU (`numpy`/`soundfile`) | ~0.5 GB RAM | **~20 – 30 giây** |
| **4. Gom cụm giọng chủ kênh (Speaker Filter)** | CPU / GPU | ~1.0 GB VRAM | **~30 – 45 giây** |
| **5. Nhận diện chữ `Faster-Whisper large-v3-turbo` (FP16)** | **RTX 3060 GPU** | **~4.5 GB VRAM** | **~4 – 6 phút** *(chỉ chạy trên ~45 phút câu đã qua 3 vòng lọc đầu)* |
| **6. Bầu chọn Golden 10s + Đóng gói `F5-TTS` / `GPT-SoVITS`** | CPU + Disk I/O | ~350 MB Disk lưu trữ cuối cùng | **~10 giây** |
| **TỔNG CỘNG (Hoàn toàn tự động)** | **RTX 3060 12GB dư sức (Peak < 6GB VRAM)** | **~350 MB / 1 mẫu giọng thành phẩm** | **⏱️ ~10 – 14 phút / 1 mẫu giọng!** |

> **Đặc biệt:** Nếu bạn chỉ cần lấy **"Golden 10s Prompt"** để chạy **Zero-shot `F5-TTS`** đọc sách ngay lập tức, hệ thống chỉ cần cào **1 video duy nhất (~10 phút)** và hoàn thành toàn bộ trong **chưa đầy 2 phút**!
