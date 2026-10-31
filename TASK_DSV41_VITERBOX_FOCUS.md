# NHIỆM VỤ TRỌNG TÂM: HOÀN THIỆN & TỐI ƯU VITERBOX ZERO-SHOT TTS (GIỌNG VUI VẺ)

## 1. ĐỊNH HƯỚNG TỪ USER
- User đã nghe thử và chốt: **"Viterbox Zero-shot TTS ngon đấy, phát triển theo hướng này đi, từ giờ chỉ tập trung hướng này"**.
- Hai vấn đề thực tế trên file `output_viterbox_tts.wav` mà user chỉ ra cần khắc phục triệt để:
  1. **"Đoạn đầu đọc nghe không ra tiếng Việt luôn":**
     - **Nguyên nhân gốc rễ đã xác định:** Câu đầu tiên của chương là `"PHẦN DẪN"` (viết IN HOA toàn bộ). Khi đưa vào `tokenizer_vi_expanded.json`, chữ `"PHẦN DẪN"` bị tách thành từng ký tự in hoa rời `['P', 'H', 'Ầ', 'N', '[SPACE]', 'D', 'Ẫ', 'N']` (IDs `[292, 284, 2507, 290, 2, 280, 2510, 290]`), khiến mô hình `T3` hiểu nhầm là từ viết tắt tiếng Anh và đọc đánh vần thành *"với hát DM nhanh"*! Trong khi nếu viết `"Phần dẫn."` hoặc `"phần dẫn."` thì tokenizer ra đúng chữ thường tiếng Việt `['P', 'h', 'ầ', 'n', '[SPACE]', 'd', 'ẫ', 'n']`.
  2. **"Đang bị dính chữ 1 chút" (đọc hơi vội, lướt chữ ở câu dài, tổng thời lượng chỉ 37.7s):**
     - **Nguyên nhân gốc rễ đã xác định:**
       - Thứ nhất: Trong `core/chatterbox_engine.py`, `repetition_penalty` đang đặt quá cao (`2.0`) và `top_p=1.0`, `cfg_weight=0.5`. Khi `repetition_penalty=2.0`, mô hình `T3` bị phạt nặng khi ngân nguyên âm, buộc nó phải nhảy âm tiết thật nhanh gây dính chữ.
       - Thứ hai: File mẫu `golden_01.wav` của kênh YouTube Vui Vẻ có tốc độ nói gốc rất nhanh (~283 từ/phút = 4.7 âm tiết/giây) và chứa từ `"SCP -610"`. Khi `prepare_conditionals` trích `t3_cond_prompt_tokens` trực tiếp từ `ref_16k_wav`, `T3` bắt chước tốc độ bắn chữ 283 WPM đó.
       - Thứ ba: Hàm `_split_text_to_sentences` trong `viterbox/tts.py` chỉ tách theo `.?!`, không tách các câu văn dài 50–65 từ có nhiều mệnh đề ngăn cách bởi dấu phẩy `,`. Khi `T3` sinh liên tục một câu 63 từ, càng về cuối câu tốc độ càng bị cuốn và dính chữ.

## 2. QUY TẮC MÔI TRƯỜNG (BẤT DI BẤT DỊCH)
- Chỉ sử dụng `.venv` hiện tại (`/home/tu/Projects/epub2m4b/.venv`, Python 3.12).
- Không tạo thêm bất kỳ virtualenv nào; không downgrade `torch`, `numpy`, `librosa`, `scipy`.
- Không gọi tool `question` để hỏi lại; chủ động thực hiện hoàn chỉnh từ Bước 1 đến Bước 5.

## 3. GIẢI PHÁP KỸ THUẬT CẦN TRIỂN KHAI (STEP 1 -> STEP 5)

### Bước 1 — Chuẩn hóa Văn bản Tiếng Việt & Chống đọc sai chữ IN HOA cho Viterbox
Trong `core/chatterbox_engine.py` (hoặc hàm tiền xử lý văn bản của `ViterboxTTSEngine`):
1. **Xử lý từ viết IN HOA tiếng Việt (Fix 100% lỗi đoạn đầu `"PHẦN DẪN"`):**
   - Chuyển đổi mọi từ viết hoa tiếng Việt (từ có >= 2 chữ cái in hoa như `PHẦN`, `DẪN`, `CHƯƠNG`, `TRƯƠNG`...) về dạng chữ thường / Sentence case chuẩn (`"PHẦN DẪN"` -> `"Phần dẫn."`).
   - Kiểm tra bằng tokenizer: Đảm bảo sau khi chuẩn hóa, chuỗi token của `"PHẦN DẪN"` chỉ chứa các chữ cái thường (`'P', 'h', 'ầ', 'n', '[SPACE]', 'd', 'ẫ', 'n', '.'`).
2. **Chia nhỏ câu dài theo mệnh đề (Clause Chunking chống dính chữ):**
   - Nếu một câu dài quá 18–22 từ và có dấu phẩy `,` (hoặc `;`, `–`), hãy tách thành các mệnh đề ngắn hợp lý (mỗi mệnh đề khoảng 8–20 từ, kết thúc bằng dấu chấm `.` khi đưa vào `_generate_single` để `T3` hạ giọng tự nhiên và phát âm rõ từng từ), sau đó ghép lại với khoảng nghỉ nhẹ `pause_ms = 220..280ms` giữa các mệnh đề phẩy và `500ms` giữa các câu chấm.
   - Với các tiêu đề ngắn (như `"Phần dẫn."` chỉ có 2–3 từ), đảm bảo không bị `vad_trim` cắt cụt đuôi âm thanh (dùng `margin_s=0.12` trở lên).

### Bước 2 — Tinh chỉnh Tham số & Nhịp độ Giọng mẫu (Pacing & Clarity Tuning)
1. **Chọn/Tối ưu Reference Prompt cho `T3` vs `S3Gen`:**
   - Trong `prepare_conditionals` (hoặc trong `ViterboxTTSEngine`):
     - Giữ nguyên sóng âm gốc `s3gen_ref_wav` cho `self.s3gen.embed_ref(...)` và `self.ve.embeds_from_wavs(...)` để **bảo toàn 100% âm sắc (timbre) trong trẻo của Vui Vẻ** (Speaker Similarity > 0.95).
     - Đối với `ref_16k_wav` đưa vào `s3_tokzr.forward` để tạo `t3_cond_prompt_tokens` (quyết định tốc độ nhịp đọc của `T3`): áp dụng `librosa.effects.time_stretch(ref_16k_wav, rate=0.82)` (làm chậm nhịp prompt cho `T3` khoảng 18%) VÀ/HOẶC chọn clip sạch thuần Việt không có chữ `"SCP"` (hoặc kết hợp cả hai).
2. **Tinh chỉnh Hyperparameters mặc định của `ViterboxTTSEngine`:**
   - `repetition_penalty`: hạ từ `2.0` xuống **`1.25`** (để nguyên âm được ngân tròn vành, không bị lướt chữ).
   - `cfg_weight`: hạ từ `0.5` xuống **`0.35`** (theo đúng tài liệu Chatterbox giúp nhịp đọc điềm đạm, rõ chữ hơn).
   - `exaggeration`: **`0.40`**
   - `temperature`: **`0.65`**
   - `top_p`: **`0.90`**
3. **Cache `TTSConds`:**
   - Cache kết quả `prepare_conditionals` theo `(audio_prompt, exaggeration)` để không phải tính lại ở mỗi segment, giúp tăng tốc độ tổng hợp chương sách.

### Bước 3 — Tích hợp `ViterboxTTSEngine` làm Engine Chính trong `core/tts_engine.py` & `app.py`
- Export `ViterboxTTSEngine` trong `core/tts_engine.py`.
- Cập nhật `app.py` (Gradio Studio) để lựa chọn mặc định / ưu tiên cho chế độ Clone giọng là **Viterbox Zero-shot TTS (Giọng Vui Vẻ)**, giúp user bấm Preview hoặc Render trên Web UI sẽ chạy trực tiếp `ViterboxTTSEngine` đã tối ưu.

### Bước 4 — Render lại Chương `# PHẦN DẪN` & Kiểm chứng bằng Faster-Whisper
- Cập nhật và chạy `scripts/render_chapter_chatterbox.py` (tập trung vào `ViterboxTTSEngine`) để xuất:
  - `workspace/dua_tre_hu_tu_kim_tran/output_viterbox_tts.wav`
  - `workspace/dua_tre_hu_tu_kim_tran/output_viterbox_tts.m4b`
- Chạy `scripts/verify_chatterbox.py` kiểm tra bằng Faster-Whisper (`small`, CUDA, `lang="vi"`):
  - **Yêu cầu bắt buộc:**
    1. Câu mở đầu `"PHẦN DẪN"` phải được Whisper nhận diện đúng là `"Phần dẫn"` (tuyệt đối không còn `"với hát DM nhanh"` hay tiếng lạ).
    2. Thời lượng chương `# PHẦN DẪN` đạt khoảng **43s – 55s** (tốc độ đọc điềm đạm, rõ chữ, không bị dính chữ).
    3. WER giảm mạnh (mục tiêu **< 15%**), Hallucination = **0%**, Speaker Similarity với Vui Vẻ **>= 0.94**.

### Bước 5 — Unit Tests & Báo cáo
- Cập nhật/bổ sung unit tests trong `tests/test_chatterbox_engine.py` (test chuẩn hóa chữ IN HOA `"PHẦN DẪN"` -> `"Phần dẫn."`, test chia mệnh đề câu dài, test caching conditionals).
- Chạy `.venv/bin/pytest -v` đảm bảo 100% tests PASS.
- Cập nhật báo cáo kết quả vào `voices/vui_ve/CHATTERBOX_REPORT.md` và `voices/vui_ve/CHATTERBOX_VERIFICATION.md`.
