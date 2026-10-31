# Design Doc: Dynamic Voice Catalogue, Golden Showcase, Adaptive Batching & Uncertainty Scanner

**Date:** 2026-10-31  
**Target:** `core/vieneu_engine.py`, `core/text_normalizer.py`, `scripts/generate_voice_showcase.py`, `tests/`

---

## 1. Mục Tiêu & Phạm Vi (Scope)

1. **Loại bỏ Hardcode Giọng Đọc (Dynamic Voice Catalogue):**
   - Xoá sạch mọi danh sách giọng tĩnh (`FALLBACK_PRESET_VOICES`, các clone profile cũ).
   - Truy vấn động trực tiếp từ `tts.list_preset_voices()` của VieNeu-TTS v3 Turbo (25 giọng chính thống).
   - Cung cấp hàm phân loại giọng đọc: Nhóm ⭐ Tuyển chọn (Top Stars), Nhóm Đọc truyện / Kể chuyện, Nhóm Tự nhiên / Tin tức.

2. **Tự Động Nhận Diện Phần Cứng (Adaptive Hardware-Aware Batch Size):**
   - Không ép cứng `batch_size = 4`. Tự động đo VRAM GPU tại thời điểm chạy:
     - CPU: `batch_size = 1`
     - GPU < 6GB VRAM: `batch_size = 2`
     - GPU 6GB – 10GB: `batch_size = 4`
     - GPU 10GB – 16GB (RTX 3060 12GB, RTX 4070): `batch_size = 8` (đạt điểm ngọt RTF 0.023x)
     - GPU $\ge$ 16GB: `batch_size = 16`
     - Ưu tiên ghi đè qua biến môi trường `TTS_BATCH_SIZE`.

3. **Kho Mẫu Giọng Golden Dataset 1 Phút (Voice Showcase Generator):**
   - Script: `scripts/generate_voice_showcase.py`.
   - Lấy kịch bản chuẩn Golden Stress-Test (~1 phút, 836 ký tự) từ `scripts/benchmark_stress_test.py`.
   - Render trước (pre-generate) cho toàn bộ 25 giọng chính thống vào `workspace/golden_samples/<voice_name>.ogg` (Opus) và `.wav`.
   - Sinh kèm file metadata `workspace/golden_samples/index.json`.

4. **Bộ Quét Điểm Nghi Vấn (Uncertainty Scanner):**
   - Trong `core/text_normalizer.py`: Bổ sung hàm `detect_uncertain_spans(text: str) -> list[dict]`.
   - Nhận diện các điểm văn bản nhập nhằng cần kiểm tra hoặc nhờ AI hỗ trợ:
     - Biểu thức toán học phức tạp (`=`, `^`, `\int`, `\sqrt`).
     - Ký hiệu hoá học lạ chưa có trong từ điển cache.
     - Viết tắt lạ in hoa toàn bộ không thuộc danh mục phổ biến.
     - Token chữ + số bất thường (`F-35`, `Zz7`, `COVID-19`).
     - Phân số nhập nhằng (`1/2`).
   - Trả về danh sách span kèm ngữ cảnh câu để hiển thị cho người dùng duyệt hoặc gửi LLM xử lý có chọn lọc.

---

## 2. Chi Tiết Kỹ Thuật

### 2.1. Adaptive Batch Size (`core/vieneu_engine.py`)
```python
def auto_detect_batch_size() -> int:
    env_bs = os.getenv("TTS_BATCH_SIZE")
    if env_bs and env_bs.isdigit():
        return max(1, int(env_bs))
    if not torch.cuda.is_available():
        return 1
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 6.0:
            return 2
        elif vram_gb < 10.0:
            return 4
        elif vram_gb < 16.0:
            return 8
        else:
            return 16
    except Exception:
        return 4
```

### 2.2. Dynamic Voice Catalogue (`core/vieneu_engine.py`)
- `get_available_preset_voices(tts=None) -> list[tuple[str, str]]`: Trả về danh sách `(label, name)` chính thống.
- `get_categorized_voices(tts=None) -> dict`: Phân loại thành:
  - `stars`: các giọng có tiền tố `⭐`
  - `story`: các giọng có từ khóa "kể chuyện", "đọc truyện"
  - `news_and_natural`: các giọng "tin tức", "tự nhiên"

### 2.3. Showcase Script (`scripts/generate_voice_showcase.py`)
- Render batch từng giọng vào `workspace/golden_samples/`.
- Dùng `soxr` / `ffmpeg` xuất file `.ogg` (Opus 48kbps mono) để Telegram stream mượt mà dưới dạng Voice Note native.
- Ghi `index.json` chứa: `[{"id": "Thien_Minh", "name": "Thiện Minh", "label": "...", "ogg_path": "...", "wav_path": "...", "duration_sec": 63.1}, ...]`.

### 2.4. Uncertainty Scanner (`core/text_normalizer.py`)
```python
@dataclass
class UncertainSpan:
    token: str
    sentence: str
    reason: str  # 'math_equation' | 'chem_formula' | 'mixed_alphanumeric' | 'unknown_abbr' | 'fraction'
    suggested_reading: str
```
- Quét nhanh bằng regex, trả về danh sách spans ngắn gọn giúp tiết kiệm 95% token và làm giao diện review trực quan.

---

## 3. Kế Hoạch Kiểm Thử
1. `tests/test_voice_catalogue.py`: Kiểm tra load dynamic voices, phân loại nhóm, auto detect batch size.
2. `tests/test_uncertainty_scanner.py`: Kiểm tra quét đúng các ca hóc búa, không false alarm trên văn xuôi bình thường.
3. Chạy toàn bộ pytest suite đảm bảo 100% tests pass.
