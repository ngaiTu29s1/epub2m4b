# Design Doc: Self-Enriching Hybrid Normalizer, Pipeline Speedup & Golden Stress-Test

**Date:** 2026-10-31  
**Target:** `core/text_normalizer.py`, `core/vieneu_engine.py`, `core/llm_normalizer.py`, `scripts/benchmark_stress_test.py`

---

## 1. Mục tiêu & Vấn đề giải quyết

1. **Xoá bỏ phiên âm bồi gượng gạo:**
   - Gỡ bỏ hoàn toàn việc ép từ mượn tiếng Anh thông dụng sang tiếng Việt bồi (`video` $\to$ `vi-đê-ô`, `camera` $\to$ `ca-mê-ra`, `online` $\to$ `ôn-lai`, `robot` $\to$ `rô-bốt`...) để tận dụng trực tiếp engine G2P song ngữ chuẩn quốc tế của `VieNeu-TTS v3 Turbo`.
2. **Bộ chuẩn hoá tự học đa tầng (Self-Enriching Hybrid Normalizer):**
   - **Tầng 1 (Rule-based Regex):** Xử lý số, ngày tháng, La Mã, đơn vị tiêu chuẩn (0ms).
   - **Tầng 2 (Persistent Cache / Dynamic Dictionary):** Lưu trữ dạng JSON tại `core/data/dictionary_cache.json` và `workspace/dictionary_cache.json`. Tra cứu $O(1)$ cho mọi từ viết tắt, công thức toán/hoá đã được học trong quá khứ.
   - **Tầng 3 (LLM Fallback Scanner):** Phát hiện câu có biểu thức toán ($x^2 + y^2 = z^2$, $\int$), công thức hoá học ($H_2SO_4, CO_2$), từ viết tắt lạ hoặc ngữ cảnh nhập nhằng mà Tầng 1 & Tầng 2 chưa có $\to$ gọi LLM (qua API OpenAI/Gemini/Local tương thích) để chuyển thành lời đọc tiếng Việt tự nhiên.
   - **Tầng 4 (Auto-Enrich):** Tự động nạp các cặp `(thuật ngữ -> phiên âm chuẩn)` do LLM sinh ra vào Persistent Cache để các lần sau chạy với tốc độ 0ms không tốn token.
3. **Tối ưu tốc độ kịch trần (GPU Pipeline Speedup):**
   - Tận dụng 8GB VRAM còn trống trên RTX 3060: Bật `batch_size=4` (hoặc `6`) trong `VieNeuEngine.synthesize_segment` và `render_chapter` để chạy batching trên CUDA Graph, đẩy tốc độ render sách 12 tiếng xuống còn 20–25 phút.
4. **Bộ dữ liệu chuẩn Golden Stress-Test (~1 phút):**
   - Tạo kịch bản thử nghiệm dài ~1 phút (`scripts/benchmark_stress_test.py`) chứa đầy đủ các ca hóc búa nhất:
     - Song ngữ Anh - Việt (Machine learning, video, camera, smartphone, online...).
     - Công thức hoá học: $H_2SO_4$, $CO_2$, $NaCl$, $KMnO_4$.
     - Toán học: $E = mc^2$, $x^2 + 2x + 1 = 0$, $3/4$, $\alpha, \beta$.
     - Viết tắt & tổ chức: TP.HCM, UNESCO, WHO, COVID-19, ADN, GDP.
     - Đơn vị đo lường: $3.5\text{ GHz}$, $60\text{ km/h}$, $37.5^\circ\text{C}$, $25\%$.
     - Số La Mã & ngày tháng: Thế kỷ XXI, ngày 30/04/1975.

---

## 2. Kiến trúc chi tiết

### 2.1. Cấu trúc Module
```text
core/
├── text_normalizer.py      # Tầng 1: Rule-based regex (đã dọn dẹp từ tiếng Anh)
├── llm_normalizer.py       # Tầng 2, 3, 4: Cache management + LLM Fallback client
└── data/
    └── dictionary_cache.json # Từ điển tự tích luỹ (pre-seeded với thuật ngữ thông dụng)
```

### 2.2. Cơ chế Fallback & Auto-Enrich (`core/llm_normalizer.py`)
- Mặc định đọc cấu hình từ biến môi trường (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`). Hỗ trợ chuẩn OpenAI-compatible (dùng được cả Gemini, OpenAI, Claude, Groq hoặc Ollama local).
- Nếu không có API key / offline: Graceful degradation — hệ thống tự động fallback về rule-based regex tốt nhất có thể, tuyệt đối không làm crash pipeline tạo sách.
- Khi LLM trả về bản dịch văn nói, phân tích trích xuất từ khóa mới $\to$ lưu vào `dictionary_cache.json`.

### 2.3. Tối ưu Batching trong `core/vieneu_engine.py`
- Mặc định đặt `batch_size=4` khi chạy trên thiết bị `cuda`.
- Cho phép tuỳ biến `batch_size` qua tham số khởi tạo `VieNeuEngine(batch_size=4)`.

---

## 3. Kế hoạch QA/QC & Verification
1. `tests/test_text_normalizer.py`:
   - Xác nhận các từ tiếng Anh (`video`, `camera`, `online`) được giữ nguyên chữ gốc, không bị biến thành `vi-đê-ô`.
   - Kiểm tra tra cứu cache và phát hiện các biểu thức toán/hoá.
2. `tests/test_vieneu_engine.py`:
   - Xác nhận `batch_size=4` hoạt động ổn định trên GPU, không tràn VRAM.
3. `scripts/benchmark_stress_test.py`:
   - Chạy render toàn bộ đoạn Golden Stress-Test ~1 phút.
   - Đo thời gian render thực tế, kiểm tra âm thanh đầu ra và phiên âm bằng Whisper `medium`.
