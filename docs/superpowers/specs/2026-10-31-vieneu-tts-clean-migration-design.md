# PRD & Technical Spec: Clean Migration to VieNeu-TTS v3 (Preset Voices, Nguyễn Ngọc Ngạn & Instant Clone)

**Date:** 2026-10-31  
**Project:** `epub2m4b`  
**Target Branch:** `feat/mvp`  
**OpenCode Session:** `ses_e6fad7311ffeeDj17ehf5DWyps` (`omniroute/deepseek-v4-flash`)

---

## 1. Executive Summary & User Objective

Người dùng quyết định **dọn sạch (clean up) toàn bộ các kiến trúc TTS thử nghiệm cũ (F5-TTS, RVC, Chatterbox/Viterbox)** vốn phức tạp và over-engineered, để chuyển sang sử dụng trực tiếp **`VieNeu-TTS` (`vieneu` v3.8.3 / `pnnbao-ump/VieNeu-TTS-v3-Turbo` & bản fine-tune Nguyễn Ngọc Ngạn)** làm pipeline TTS chính thức duy nhất (cùng với `Edge-TTS` dự phòng).

### Ràng buộc bắt buộc
1. **Chỉ dùng duy nhất `.venv` hiện tại** (`/home/tu/Projects/epub2m4b/.venv`, Python 3.12, `torch 2.14.1+cu130`). Khi cài `vieneu` và `neucodec` (hoặc `sea-g2p`, `soxr`, `kaldi-native-fbank`), **tuyệt đối không để pip đụng vào `torch`/`torchaudio` hay `gradio` hiện tại**.
2. **Dọn sạch code & tài nguyên kiến trúc cũ:**
   - Xoá các module/script cũ không còn dùng: `core/rvc_engine.py`, `core/chatterbox_engine.py`, `vendor/viterbox_src/`, `models/chatterbox_vui_ve/`, `models/rvc_vui_ve/`, `scripts/render_chapter_chatterbox.py`, `scripts/verify_chatterbox.py`, `scripts/check_tail_artifacts.py`, `scripts/check_pause_alignment.py`, `tests/test_rvc_engine.py`, `tests/test_chatterbox_engine.py`, và gỡ `F5TTSEngine` / `RVC` khỏi `core/tts_engine.py` & `app.py`.
3. **Đảm bảo xử lý văn bản hoàn hảo (Số, La Mã, Ngày tháng, Viết tắt) & QA/QC bằng Whisper `medium`:**
   - Dù `sea-g2p` của `vieneu` đã có G2P tốt, `core/text_normalizer.py` vẫn phải chuẩn hoá triệt để `"Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi."` thành chữ tiếng Việt rõ ràng để 100% không bao giờ đọc sai số La Mã hay ngày tháng.

---

## 2. Architecture: `core/vieneu_engine.py` & `core/tts_engine.py`

### 2.1. Cài đặt gói `vieneu` trong `.venv`
- Cài đặt các dependency nhẹ cần thiết vào `.venv`:
  `uv pip install --python .venv/bin/python vieneu neucodec sea-g2p soxr kaldi-native-fbank onnxruntime --no-deps` (hoặc cài các gói thiếu mà không nâng/hạ `torch` và `gradio`).
- Kiểm tra `from vieneu import Vieneu` hoạt động hoàn hảo.

### 2.2. Các chế độ giọng đọc trong `VieNeuEngine` (`core/vieneu_engine.py`)
`VieNeuEngine(TTSEngine)` hỗ trợ 3 chế độ "ăn liền" cực kỳ rõ ràng:
1. **Chế độ 1 — `VieNeu-TTS Preset (25 giọng Bắc/Trung/Nam 48kHz)` (MẶC ĐỊNH):**
   - Khởi tạo `Vieneu()` (sử dụng model mặc định của `vieneu` v3).
   - Lấy danh sách giọng preset bằng `tts.list_preset_voices()` và chọn giọng qua `tts.get_preset_voice(voice_id)` (mặc định chọn giọng nam đọc truyện ấm áp như `Binh`, `Tuyen`, `XuanVinh` hoặc giọng đầu tiên trong danh sách preset).
   - Sinh âm thanh qua `tts.infer(text=..., voice=voice_data)` và xuất WAV chuẩn mono PCM_16 cho pipeline đóng gói `.m4b`.
2. **Chế độ 2 — `VieNeu-TTS Kinh Điển (Nguyễn Ngọc Ngạn Fine-tune)`:**
   - Hỗ trợ nạp giọng Nguyễn Ngọc Ngạn (qua `backbone_repo="nguyen-brat/nguyen-ngoc-ngan-vieneu-tts-fine-tune"` hoặc dùng reference audio mẫu của bác Nguyễn Ngọc Ngạn / preset tương ứng trong `vieneu`).
   - Lưu ý kiểm tra xem `Vieneu` v3.8.3 nạp `nguyen-brat/nguyen-ngoc-ngan-vieneu-tts-fine-tune` qua chế độ nào (`mode` hoặc `backbone_repo`) hoặc tải 1 đoạn golden reference chuẩn của bác Nguyễn Ngọc Ngạn (`voices/nguyen_ngoc_ngan/golden_ref.wav`) từ dataset/repo để clone/infer mượt mà 100%.
3. **Chế độ 3 — `VieNeu-TTS Instant Clone (Giọng Vui Vẻ / Custom WAV)`:**
   - Cho phép truyền `ref_audio` (mặc định trỏ tới `voices/vui_ve/golden_refs/golden_01.wav` kèm `ref_text` trong `voices/vui_ve/profile.json`, hoặc bất kỳ file WAV 3–8s nào người dùng chọn trên UI).
   - Gọi `tts.infer(text=..., ref_audio=..., ref_text=...)`.

### 2.3. Chuẩn hoá Văn bản Ngữ cảnh trong `core/text_normalizer.py`
- Nâng cấp `normalize_vi_text(text: str) -> str`:
  - Chuyển đổi số nguyên/số thứ tự sang chữ tiếng Việt chuẩn miền Bắc (`number_to_words_vi`: `"nghìn"`, `"linh"`, `"mốt"`, `"tư"`, `"lăm"`).
  - Ngày tháng năm: `"Ngày 19/05/1990"` $\to$ `"Ngày mười chín tháng năm năm một nghìn chín trăm chín mươi"` (không lặp chữ `"Ngày ngày"`), `"tháng 4"` $\to$ `"tháng tư"`, `"thứ 4"` $\to$ `"thứ tư"`.
  - Số La Mã (`Chương IV` $\to$ `"Chương bốn"`, `thế kỷ XXI` $\to$ `"thế kỷ hai mươi mốt"`, đề mục `I.`, `II.`).
  - Đơn vị & viết tắt: `TP.HCM` $\to$ `"Thành phố Hồ Chí Minh"`, `1m76` $\to$ `"một mét bảy mươi sáu"`, `60cm`, `25%`, `1,5`.
  - Trong `markdown_to_segments`: đặt `max_chars=500` để câu văn được đọc liền mạch tự nhiên theo từng câu trọn vẹn, không cắt vụn giữa câu.

### 2.4. Giao diện Gradio Studio (`app.py`)
- Tinh gọn khu vực chọn Engine & Giọng đọc:
  - Engine mặc định: **`VieNeu-TTS (Giọng Tự Nhiên / Sách Nói AI)`**
  - Dropdown chọn Giọng (`voice_choice`) hiển thị trực tiếp:
    - Danh sách toàn bộ các giọng Preset của `VieNeu-TTS` (ví dụ: `Tuyên (Nam miền Bắc)`, `Bình (Nam miền Bắc)`, `Xuân Vĩnh (Nam miền Nam)`, `Nguyên`, `Sơn`...)
    - Tuỳ chọn **`Nguyễn Ngọc Ngạn (Kể chuyện kinh điển)`**
    - Tuỳ chọn **`Vui Vẻ (Clone từ kho giọng đã cào)`**
    - Tuỳ chọn **`Custom Clone (Tải file mẫu 3-8s)`**
  - Giữ lại **`Edge-TTS (Siêu tốc ऑनलाइन)`** làm phương án phụ nếu người dùng muốn render nhanh bằng giọng Microsoft.
  - Xoá bỏ hoàn toàn các ô nhập đường dẫn `.pth`/`.index` của RVC và ô F5-TTS cũ cho giao diện gọn đẹp.

---

## 3. QA/QC & Verification Plan (Dùng Whisper `medium`)

1. **Tạo script `scripts/verify_vieneu.py`:**
   - Render câu mẫu: `"Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi."` ra `workspace/_preview/sample_vieneu.wav`.
   - Render chương `# PHẦN DẪN` của cuốn *Đứa Trẻ Hư* ra `workspace/dua_tre_hu_tu_kim_tran/output_vieneu_tts.wav` và đóng gói `.m4b` (`output_vieneu_tts.m4b`).
   - Dùng **`faster-whisper` model `medium`** (tuyệt đối không dùng `small`) để chấm WER:
     - Câu mẫu `"Chương IV: Ngày 19/05/1990 tại TP.HCM..."` phải đọc đúng 100% số La Mã, ngày tháng năm, viết tắt (WER $\le 5\%$).
     - Chương `# PHẦN DẪN` phải đọc đúng mở đầu `"Phần dẫn"`, WER trên Whisper `medium` $< 10\%$, nhịp điệu tự nhiên, âm thanh sạch không rè đuôi.
2. **Unit Tests (`tests/`):**
   - Cập nhật `tests/test_text_normalizer.py`, `tests/test_tts_engine.py`, `tests/test_app_ui.py` và thêm `tests/test_vieneu_engine.py`.
   - Chạy `.venv/bin/pytest -q` đạt 100% PASS.
