# NHIỆM VỤ DỰ ÁN EPUB2M4B: TÍCH HỢP CHATTERBOX / VITERBOX-TTS & CHATTERBOX-VC CHO GIỌNG VUI VẺ

## 1. BỐI CẢNH & QUY TẮC BẮT BUỘC
- **Dự án:** `epub2m4b` tại `/home/tu/Projects/epub2m4b` (branch: `feat/mvp`).
- **Môi trường bắt buộc:** DUY NHẤT môi trường ảo `/home/tu/Projects/epub2m4b/.venv` (Python 3.12, `torch 2.14.1+cu130`).
  - ⚠️ **QUY TẮC NGHIÊM NGẶT:**
    1. **TUYỆT ĐỐI KHÔNG TẠO THÊM BẤT KỲ VENV NÀO** (không tạo `.venv-*`, không dùng conda).
    2. **KHÔNG HẠ CẤP (DOWNGRADE)** `torch`, `torchaudio`, `numpy`, `librosa`, `scipy` đang có trong `.venv`.
    3. Khi cài đặt thư viện mới vào `.venv`, dùng `uv pip install --python .venv/bin/python --no-deps ...` cho các package có pin cứng version cũ, và cài riêng các dependency thuần cần thiết (`s3tokenizer`, `conformer`, `diffusers`, `resemble-perth`, `omegaconf`, `einops`, `soe-vinorm`, `pykakasi`).

## 2. MỤC TIÊU KỸ THUẬT
Trước đó:
- F5-TTS zero-shot bị hallucination câu outro YouTube.
- RVC v2 base model (`f0G40k.pth`) bị hiện tượng phát âm "lưỡi ngắn / đớt" do generator base pre-train trên tiếng Trung/Anh.

User yêu cầu tích hợp hệ sinh thái **`resemble-ai/chatterbox`** kết hợp bản fine-tune tiếng Việt 3.000 giờ **`dolly-vn/viterbox`** (GitHub: `https://github.com/iamdinhthuan/viterbox-tts`, HuggingFace: `dolly-vn/viterbox`) để giải quyết triệt để cả hai vấn đề trên với 2 chế độ:

### Chế độ 1: `ViterboxTTSEngine` (Zero-shot Vietnamese TTS + Voice Cloning)
- Sử dụng mã nguồn `viterbox` (`https://github.com/iamdinhthuan/viterbox-tts`) và trọng số từ HuggingFace `dolly-vn/viterbox` (`t3_ml24ls_v2.safetensors`, `s3gen.pt`, `ve.pt`, `tokenizer_vi_expanded.json`, `conds.pt`).
- Dùng file mẫu vàng của kênh Vui Vẻ: `voices/vui_ve/golden_refs/golden_01.wav` (dài 9.1s, chuẩn 24kHz mono) làm `audio_prompt` để clone giọng Vui Vẻ đọc trực tiếp tiếng Việt.
- Tích hợp vào `core/chatterbox_engine.py` (và expose qua `core/tts_engine.py`) tuân thủ interface `TTSEngine.synthesize_segment(text: str, out_wav: Path) -> int` (xuất WAV 24000Hz, mono, PCM_16).

### Chế độ 2: `ChatterboxVCEngine` (Edge-TTS + Chatterbox S3Gen Voice Conversion)
- Sử dụng cơ chế `ChatterboxVC` (`src/chatterbox/vc.py` từ `resemble-ai/chatterbox`), nạp bộ giải mã `S3Gen` từ `dolly-vn/viterbox` (`s3gen.pt` đã được fine-tune trên 3.000h tiếng Việt).
- Quy trình:
  1. Dùng `EdgeTTSEngine` (`vi-VN-NamMinhNeural`) đọc văn bản tiếng Việt ra file WAV tạm (đảm bảo phát âm từ vựng chuẩn 100%, tốc độ chuẩn).
  2. Đưa file WAV qua `S3Gen` (`s3gen.tokenizer` -> `s3gen.inference` với `ref_dict` trích xuất từ `voices/vui_ve/golden_refs/golden_01.wav`) để chuyển đổi âm sắc sang giọng Vui Vẻ ở tần số 24000Hz mono PCM_16.

## 3. LƯU Ý TƯƠNG THÍCH PYTHON 3.12 & TRANSFORMERS TRONG `.venv`
- Kiểm tra kỹ tương thích của `viterbox` với `transformers` hiện có trong `.venv`:
  - Nếu `transformers 5.x` đổi vị trí import `TopPLogitsWarper` hoặc `LlamaModel` / `_attn_implementation` (ví dụ `sdpa` vs `eager`), hãy patch trực tiếp trong module hoặc vendored code của `viterbox` để chạy mượt trên `.venv` hiện tại mà không cần downgrade `transformers` nếu gây xung đột.
  - Nếu `spacy-pkuseg` (chỉ dùng cho tiếng Trung) lỗi build Cython trên Python 3.12, hãy bỏ qua không cài `spacy-pkuseg` và làm nhẹ phần tokenizer tiếng Trung trong `tokenizers.py` vì ta chỉ dùng tiếng Việt (`vi`) và tiếng Anh (`en`).
  - Nếu `resemble-perth` không tương thích hoặc không cần thiết, có thể dùng trực tiếp sóng âm raw từ `s3gen` mà không cần đóng watermark.

## 4. CÁC BƯỚC THỰC HIỆN CỤ THỂ (STEP 1 -> STEP 5)
1. **Bước 1 — Cài đặt & Chuẩn bị trong `.venv`:**
   - Clone hoặc tải mã nguồn `iamdinhthuan/viterbox-tts` (và `vc.py` của `resemble-ai/chatterbox`) vào thư mục nội bộ hoặc cài editable `--no-deps` vào `.venv`.
   - Cài đặt các dependency phụ trợ còn thiếu (`s3tokenizer`, `conformer`, `diffusers`, `einops`, `omegaconf`, `soe-vinorm`, ...) vào `.venv` mà **không làm thay đổi** `torch 2.14.1+cu130` và `numpy`.
   - Tải checkpoint `dolly-vn/viterbox` từ HuggingFace.
2. **Bước 2 — Xây dựng `core/chatterbox_engine.py` & Unit Tests (`tests/test_chatterbox_engine.py`):**
   - Cài đặt `ViterboxTTSEngine(TTSEngine)` (Zero-shot TTS giọng Vui Vẻ).
   - Cài đặt `ChatterboxVCEngine(TTSEngine)` (Edge-TTS + S3Gen VC giọng Vui Vẻ).
   - Viết đầy đủ unit tests trong `tests/test_chatterbox_engine.py`.
3. **Bước 3 — Render thực nghiệm chương `# PHẦN DẪN` của *"Đứa Trẻ Hư"* (`workspace/dua_tre_hu_tu_kim_tran/book.md`):**
   - Tạo script `scripts/render_chapter_chatterbox.py` để render chương `# PHẦN DẪN` bằng cả 2 chế độ:
     - Output 1 (Viterbox TTS): `workspace/dua_tre_hu_tu_kim_tran/output_viterbox_tts.wav` & `.m4b`
     - Output 2 (Edge-TTS + ChatterboxVC): `workspace/dua_tre_hu_tu_kim_tran/output_chatterbox_vc.wav` & `.m4b`
4. **Bước 4 — Kiểm chứng chất lượng bằng Faster-Whisper & Cosine Similarity:**
   - Transcribe cả 2 file audio vừa tạo bằng Faster-Whisper (`small`, `lang="vi"`).
   - Đo WER (Word Error Rate), kiểm tra 0% hallucination, đo thời lượng (duration hợp lý ~45-60s, không bị tua nhanh), và đo độ tương đồng giọng (Speaker Similarity) so với `voices/vui_ve/golden_refs/golden_01.wav`.
5. **Bước 5 — Chạy toàn bộ Test Suite & Xuất báo cáo:**
   - Chạy `.venv/bin/pytest -v` đảm bảo toàn bộ tests (cũ + mới) PASS 100%.
   - Ghi báo cáo chi tiết kết quả so sánh vào `voices/vui_ve/CHATTERBOX_REPORT.md`.
