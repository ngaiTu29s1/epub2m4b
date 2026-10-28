# NHIỆM VỤ DỰ ÁN EPUB2M4B: TRIỂN KHAI PHƯƠNG ÁN A (EDGE-TTS + RVC V2 CLONE GIỌNG VUI VẺ)

## 1. BỐI CẢNH & YÊU CẦU CỐT LÕI
- **Dự án:** `epub2m4b` tại `/home/tu/Projects/epub2m4b` (branch: `feat/mvp`).
- **Môi trường bắt buộc:** DUY NHẤT môi trường ảo `.venv` hiện tại của repo (`/home/tu/Projects/epub2m4b/.venv`, Python 3.12).
  - ⚠️ **QUY TẮC BẤT DI BẤT DỊCH TỪ USER:** TUYỆT ĐỐI KHÔNG TỰ Ý TẠO THÊM BẤT KỲ VENV MỚI NÀO (không tạo `.venv-py310`, không dùng conda ngoài luồng). Mọi package nếu cần cài thêm phải được cài trực tiếp vào `.venv` hiện tại.
- **Dataset giọng Vui Vẻ đã có sẵn:**
  - 226 audio clips vocal phòng thu chuẩn tại `voices/vui_ve/wavs/` (~26.67 phút, mono 24000Hz, PCM_16).
  - Metadata: `voices/vui_ve/metadata.csv`, `voices/vui_ve/profile.json`.

## 2. VẤN ĐỀ CẦN GIẢI QUYẾT
- F5-TTS Zero-shot hiện tại trên tiếng Việt (`toandev/F5-TTS-Vietnamese`) bị lỗi catastrophic hallucination nặng (lặp lại câu outro YouTube: *"Hãy subscribe cho kênh La La School / Ghiền Mì Gõ"*), không đọc đúng nội dung sách.
- User đã chốt chuyển sang **Phương án A: Edge-TTS + RVC v2**:
  - Dùng **Edge-TTS** (giọng nam miền Bắc `vi-VN-NamMinhNeural` hoặc `vi-VN-HoaiMyNeural` đã có sẵn và chạy cực tốt) để sinh vocal tiếng Việt chuẩn xác 100%, không bị ảo giác.
  - Áp dụng **Voice Conversion (RVC v2)** để đổi âm sắc (timbre) từ giọng Edge-TTS sang chất giọng đặc trưng của kênh **Vui Vẻ** dựa trên 226 audio clips trong `voices/vui_ve/wavs/`.

## 3. LƯU Ý KỸ THUẬT VỀ RVC TRÊN PYTHON 3.12
- Thư viện cũ `rvc-python==0.0.6` đòi hỏi `fairseq==0.12.2` và `numpy<1.26` vốn bị lỗi compile do Python 3.12 bỏ `distutils`.
- Bạn hãy triển khai RVC inference/conversion tương thích Python 3.12:
  - Có thể sử dụng implementation RVC inference thuần PyTorch / ONNX Runtime (như Applio core inference, hoặc direct HubERT feature extraction + VITS net_g) hoặc cài wheel tương thích đã build cho torch hiện tại trong `.venv`.
  - Cần pre-trained contentvec / HuBERT base checkpoint và pre-trained RVC v2 base generator nếu cần train / index, hoặc trích xuất feature index FAISS từ 226 clips `voices/vui_ve/wavs/`.

## 4. QUY TRÌNH THỰC HIỆN CỤ THỂ
1. **Kiểm tra môi trường:**
   - Dùng `.venv/bin/python` để kiểm tra các package hiện có (`torch`, `torchaudio`, `edge-tts`, `soundfile`, `scipy`, ...).
2. **Xây dựng module RVC Voice Conversion:**
   - Tạo/cập nhật module voice conversion (ví dụ `core/rvc_engine.py` hoặc tích hợp vào `core/tts_engine.py`) để:
     - Input: Audio từ Edge-TTS (`vi-VN-NamMinhNeural`).
     - Output: Audio đã được morph sang chất giọng Vui Vẻ, sample rate 24000Hz mono PCM_16.
3. **Thực nghiệm Render toàn bộ chương mẫu:**
   - Sách mẫu: `workspace/dua_tre_hu_tu_kim_tran/book.md` (chương `# PHẦN DẪN`).
   - Tạo audio hoàn chỉnh cho chương này và lưu tại `workspace/dua_tre_hu_tu_kim_tran/output_chapter_vui_ve.m4b` (hoặc `.wav`).
4. **Kiểm chứng chất lượng (Verification):**
   - Dùng Faster-Whisper (`scripts/verify_chapter_dua_tre_hu.py` hoặc tương đương) để transcribe lại audio vừa tạo.
   - So sánh transcription với text gốc để chứng minh: **0% hallucination, 100% tiếng Việt chuẩn xác, mang âm sắc Vui Vẻ**.
5. **Kiểm tra Unit Tests & Báo cáo:**
   - Chạy `.venv/bin/pytest -v` đảm bảo toàn bộ tests hiện có vẫn PASS 100%.
   - Xuất file báo cáo chi tiết vào `voices/vui_ve/RVC_REPORT.md`.

Hãy tiến hành từng bước cẩn thận, tự kiểm tra mã nguồn và kết quả trước khi hoàn thành!
