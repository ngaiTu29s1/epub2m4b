# Design Doc: Audiobookshelf Integration & WebUI Studio Polish

**Date:** 2026-11-01  
**Target:** `core/abs_client.py`, `app.py`, `.env.example`, `.env`, `tests/test_abs_client.py`, `tests/test_ui_polish.py`

---

## 1. Bối Cảnh & Mục Tiêu

### 1.1. Bối Cảnh Triển Khai Thực Tế
- **Máy Trạm PC (RTX 3060 12GB):**
  - Đóng vai trò "nhà máy sản xuất" audio. Bật khi cần tạo sách.
  - Tốc độ render cực nhanh: RTF 0.032× (cuốn sách 11 tiếng render xong trong ~21 phút).
  - WebUI Gradio chạy local trên máy này.
- **Server Homelab (Intel Core i3-6100):**
  - Chạy 24/7 trong mạng LAN, host Audiobookshelf (ABS) phục vụ app nghe sách trên điện thoại/tablet/web.
  - Phần cứng không thích hợp render neural TTS nặng, nhưng quản lý thư viện và stream audio hoàn hảo.
- **Luồng Tự Động Hóa 1-Click:**
  User thả EPUB vào WebUI trên PC -> Review kịch bản & điểm nghi vấn -> Chọn giọng nghe thử ngay (0ms delay) -> Bấm *"Tạo Sách & Đẩy Thẳng Sang Audiobookshelf"* -> PC render M4B kèm cover/timeline -> Tự động upload/copy sang server ABS -> Trigger ABS scan -> App điện thoại rung chuông có sách mới ngay lập tức!

### 1.2. Mục Tiêu Cụ Thể
1. **Module `core/abs_client.py`:**
   - Client giao tiếp API Audiobookshelf chuẩn REST.
   - Hỗ trợ cả 2 phương thức:
     - **Upload qua HTTP multipart:** `POST /api/libraries/{id}/upload` (phù hợp khi không mount ổ đĩa).
     - **Copy trực tiếp qua thư mục mount (SMB/NFS):** `{storage_path}/{author}/{title}/{title}.m4b` (siêu nhanh trong mạng LAN gigabit).
   - Tự động phát hiện thư viện sách nói nếu user không chỉ định `ABS_LIBRARY_ID`.
   - Gọi `POST /api/libraries/{id}/scan` để ABS index ngay sách mới.
   - Kiểm tra kết nối (`check_connection`), bắt lỗi an toàn (graceful failure, không làm crash luồng render M4B).
2. **Cấu Hình Môi Trường (`.env.example` & `.env`):**
   - Đầy đủ placeholder cho LLM Contextual Normalizer, Audiobookshelf, TTS Hardware options.
   - Hàm tiện ích tự động đọc file `.env` nếu có mà không phụ thuộc thư viện ngoài nặng nề.
3. **Tuốt Tát WebUI Gradio (`app.py`):**
   - **Kho Mẫu Giọng 25 Giọng (0ms Delay Showcase):**
     - Nạp dữ liệu từ `workspace/golden_samples/index.json`.
     - Tự động nạp file `.ogg` / `.wav` vào Audio Player khi đổi giọng trong dropdown, hiển thị thời lượng và phân nhóm (Kể chuyện, Tuyển chọn, Tin tức).
   - **Bảng "Điểm Nghi Vấn" (Uncertainty Review View):**
     - Tích hợp `detect_uncertain_spans()` khi phân tích EPUB.
     - Hiển thị bảng tương tác các từ lạ/toán/hoá/viết tắt để người dùng soát nhanh trong 30 giây trước khi bắt đầu render.
   - **Cấu hình & Tích Hợp Audiobookshelf:**
     - Accordion cấu hình ABS (URL, Token, Thư viện, nút Test Connection).
     - Checkbox "Tự động đẩy sang Audiobookshelf sau khi render".
     - Trạng thái đẩy sách & direct link sau khi render xong.
4. **Kiểm Thử & Mock:**
   - Bộ unit tests mock HTTP cho toàn bộ các method của `ABSClient`.
   - Test UI handlers cho showcase audio, uncertainty parser và ABS sync callback.

---

## 2. Thiết Kế Chi Tiết

### 2.1. Cấu Trúc Module `core/abs_client.py`

```python
class ABSClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        library_id: str | None = None,
        storage_path: str | Path | None = None,
        auto_scan: bool = True,
        timeout: float = 30.0,
    ):
        ...

    def is_configured(self) -> bool:
        """Kiểm tra xem client đã được cấu hình tối thiểu URL và Token hay chưa."""

    def check_connection(self) -> dict:
        """
        Gọi GET /api/me hoặc GET /api/libraries để kiểm tra kết nối.
        Trả về: {'connected': bool, 'username': str, 'server_version': str, 'error': str | None}
        """

    def get_libraries(self) -> list[dict]:
        """Lấy danh sách các thư viện trong ABS."""

    def find_audiobook_library_id(self) -> str | None:
        """Tìm ID của thư viện đầu tiên có mediaType == 'book'."""

    def upload_file(self, file_path: Path, library_id: str | None = None) -> dict:
        """
        Upload file M4B qua multipart/form-data lên ABS.
        Endpoint: POST /api/libraries/{id}/upload
        """

    def copy_to_storage(
        self, file_path: Path, author: str, title: str, storage_path: Path | None = None
    ) -> Path:
        """
        Copy file M4B trực tiếp vào thư mục mount SMB/NFS theo chuẩn:
        {storage_path}/{author}/{title}/{title}.m4b
        """

    def trigger_scan(self, library_id: str | None = None) -> dict:
        """
        Yêu cầu ABS scan thư viện để nhận sách mới.
        Endpoint: POST /api/libraries/{id}/scan
        """

    def sync_audiobook(
        self,
        file_path: Path,
        author: str,
        title: str,
        library_id: str | None = None,
        auto_scan: bool | None = None,
    ) -> dict:
        """
        Phương thức đồng bộ toàn diện:
        1. Nếu storage_path hợp lệ và tồn tại -> Copy file trực tiếp.
        2. Nếu không -> Upload qua REST API HTTP multipart.
        3. Nếu auto_scan = True -> Kích hoạt scan thư viện.
        """
```

### 2.2. Đọc Config An Toàn (`core/config.py` hoặc nhúng trong `core/abs_client.py`)
- Hỗ trợ đọc cả `os.environ` và file `.env` ở thư mục gốc dự án.
- Parse key-value đơn giản, an toàn, strip quotes.

### 2.3. Tích Hợp WebUI Gradio (`app.py`)
1. **Voice Showcase Player:**
   - Dropdown chọn giọng kích hoạt event `.change()` -> trả về filepath của file `.ogg` hoặc `.wav` trong `workspace/golden_samples/`.
   - Audio component dạng `autoplay=False`, cho phép bấm Play nghe lập tức không cần render lại.
2. **Uncertainty Review Component:**
   - Trong Tab 1 (Nhập sách & Kịch bản), bổ sung Accordion *"🔍 Điểm nghi vấn cần soát duyệt (Uncertainty Review)"*.
   - Hiển thị DataFrame gồm 4 cột: `Từ/Ký hiệu`, `Phân loại`, `Ngữ cảnh trong câu`, `Đề xuất phát âm`.
   - Cho phép người dùng yên tâm rằng các công thức lạ đã được phát hiện trước khi TTS chạy.
3. **Audiobookshelf Settings & One-Click Trigger:**
   - Accordion *"📚 Tích hợp Audiobookshelf (Server Homelab)"*.
   - Cho phép đổi URL/Token ngay trên giao diện hoặc lưu cấu hình.
   - Nút *"Kiểm tra kết nối"* cập nhật nhãn trạng thái xanh lá/đỏ.
   - Checkbox *"Tự động đồng bộ sang Audiobookshelf sau khi đóng gói M4B"* (mặc định bật).
   - Sau khi đóng gói M4B xong:
     - Tự động gọi `sync_audiobook()`.
     - Xuất log chi tiết kèm thông báo thành công: `✅ Đã đẩy thành công sách lên Audiobookshelf!`.

---

## 3. Kế Hoạch Kiểm Thử
1. `tests/test_abs_client.py`:
   - Test kết nối thành công / thất bại (mock requests).
   - Test auto-detect thư viện khi `library_id` rỗng.
   - Test upload multipart HTTP.
   - Test copy trực tiếp SMB/NFS khi `storage_path` được chỉ định.
   - Test trigger scan.
   - Test sync_audiobook cả hai luồng (copy vs upload) và fallback an toàn.
2. `tests/test_ui_polish.py`:
   - Test helper lấy đường dẫn sample voice showcase từ `index.json`.
   - Test helper quét uncertainty span cho toàn bộ các chương đã chọn.
   - Test callback ABS test connection và ABS sync.
3. Chạy toàn bộ pytest suite (230+ tests) đảm bảo 100% pass và không gây regression.
