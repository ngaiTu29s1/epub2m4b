# 📘 Hướng dẫn Sử dụng Chi tiết (`epub2m4b`)

Hướng dẫn này giúp bạn sử dụng thành thạo giao diện **EPUB to M4B Audiobook Studio**, tùy chỉnh kịch bản Markdown, cấu hình giọng đọc AI trên GPU **RTX 3060 12GB** và đưa sách lên **Audiobookshelf**.

---

## 1. Khởi chạy Web UI Studio

Từ thư mục gốc `/home/tu/Projects/epub2m4b`, chạy lệnh:

```bash
.venv/bin/python app.py
```

Sau đó mở trình duyệt trên Windows (hoặc Linux) truy cập:
👉 **`http://localhost:7860`**

---

## 2. Quy trình 3 bước tạo Sách Nói `.M4B`

### Bước 1: Tải sách `.epub` & Hiệu đính kịch bản Clean Markdown (`.md`)
1. **Kéo thả file `.epub`** vào ô **"Kéo thả file .epub vào đây"**.
2. Hệ thống sẽ tự động trích xuất:
   - **Ảnh bìa sách**, **Tên sách**, **Tác giả** (bạn có thể sửa lại tên sách/tác giả nếu metadata gốc bị sai).
   - **Danh sách chương:** Các mục rác (như *Trang đầu*, *Mục lục*, *Bản quyền*) đã được tự động **bỏ tích**. Các chương nội dung chính (`GIỚI THIỆU`, `PHẦN DẪN`, `CHƯƠNG 1`..`CHƯƠNG N`) được tích chọn đầy đủ.
   - **Khung soạn thảo Clean Markdown (`.md`):** Hiển thị toàn bộ văn bản sạch sẽ sẽ được đọc.
3. **Mẹo biên tập trực tiếp trong khung Clean Markdown:**
   - Dòng bắt đầu bằng `# Tên chương`: Đánh dấu **bắt đầu một chương mới** trên thanh Chapter Timeline của `.m4b` (và nghỉ `2 giây` sau khi đọc tên chương).
   - Dòng bắt đầu bằng `## Tiêu đề phụ`: Đánh dấu mục nhỏ bên trong chương (nghỉ `1.2 giây`).
   - Dòng chỉ có `---`: Đánh dấu **chuyển cảnh** (tăng nhịp nghỉ lên `1.5 giây`).
   - Nếu chỉ muốn **test nhanh 1–2 chương đầu**, bạn có thể bỏ tích các chương sau trong danh sách chương (hoặc xóa bớt trong ô Markdown) trước khi bấm tạo sách nói.

---

### Bước 2: Cấu hình Giọng đọc (TTS Engine) & Nghe thử (Preview)

1. **Chế độ 1 – `Edge-TTS (Siêu nhanh)` *(Khuyên dùng mặc định)*:**
   - Chọn giọng cơ sở:
     - `vi-VN-NamMinhNeural` (Giọng Nam trầm ấm, hợp truyện trinh thám, lịch sử, kỹ năng).
     - `vi-VN-HoaiMyNeural` (Giọng Nữ truyền cảm, hợp tiểu thuyết, tản văn).
   - Chỉnh **Tốc độ đọc (`%`)**: Mặc định `0%` (tăng lên `+5%` hoặc `+10%` nếu muốn nhịp đọc nhanh hơn).
   - Bấm **"🔊 Đọc thử (Preview)"** để nghe ngay câu mẫu trên trình duyệt.

2. **Chế độ 2 – `Edge-TTS + RVC (Đổi màu giọng RTX 3060)`:**
   - Phù hợp khi bạn muốn cách ngắt nghỉ và phát âm chuẩn Tiếng Việt của Microsoft nhưng đổi sang chất giọng của MC/diễn viên yêu thích (qua model RVC v2 `.pth` + `.index`).
   - Cài đặt thêm thư viện RVC vào `.venv` (khi cần dùng):
     ```bash
     uv pip install rvc-python
     ```
   - Mở mục **"Cấu hình nâng cao (RVC v2 / F5-TTS GPU RTX 3060)"**, điền đường dẫn tuyệt đối tới file `.pth` và `.index`.

3. **Chế độ 3 – `F5-TTS Clone (Nhân bản giọng RTX 3060)`:**
   - Dùng khi bạn có một đoạn thu âm mẫu dài **10–15 giây** (không nhạc nền, rõ chữ) và muốn nhân bản giọng đó trên GPU RTX 3060 12GB.
   - Đảm bảo lệnh `f5-tts_infer-cli` đã có trong môi trường, tải file audio mẫu 10–15s lên ô **Audio mẫu F5-TTS** và nhập chính xác nội dung câu nói trong audio mẫu vào ô **Transcript**.

---

### Bước 3: Đóng gói `.M4B` & Kiểm tra Chapter Timeline

1. Bấm nút **"🚀 Tạo Sách Nói (.M4B)"**.
2. Thanh tiến độ sẽ hiển thị từng chương đang được tổng hợp (`Đang tổng hợp (X/N): CHƯƠNG X`).
   - **Cơ chế Resume/Checkpoint tự động:** Mỗi chương hoàn tất đều được lưu ngay tại `workspace/<ten_sach>/chapters_audio/chapter_XXX.wav` và `.json`. Nếu bạn tắt máy giữa chừng, lần sau bấm tạo lại (giữ nguyên ô *"Render lại toàn bộ"* ở trạng thái **bỏ tích**) hệ thống sẽ đi tiếp từ chương đang dở.
3. Khi hoàn tất:
   - **Bảng Chapter Timeline (`FFMETADATA`):** Hiển thị rõ `Chương`, `Bắt đầu`, `Kết thúc`, `Thời lượng`.
   - **Trình phát `.m4b`** cho phép nghe trực tiếp trên trình duyệt.
   - Tải về 2 file thành phẩm: **`.m4b`** và **`.vtt`** (hoặc lấy trực tiếp trong thư mục `workspace/<ten_sach>/`).

---

## 3. Đưa sách lên máy chủ Audiobookshelf

Để **Audiobookshelf** hiển thị đầy đủ ảnh bìa, danh sách chương (Chapter Timeline) và phụ đề chạy chữ (Subtitles):

1. Tạo một thư mục riêng cho cuốn sách trong thư viện Audiobookshelf của bạn, ví dụ:
   ```text
   /audiobooks/Tử Kim Trần/Đứa Trẻ Hư/
   ├── dua_tre_hu.m4b
   └── dua_tre_hu.vtt
   ```
2. Đặt cả 2 file `.m4b` và `.vtt` **cùng tên gốc** (`dua_tre_hu.m4b` và `dua_tre_hu.vtt`) trong cùng thư mục.
3. Trên giao diện **Audiobookshelf**, bấm **Scan Library**:
   - Ảnh bìa, Tên sách, Tác giả và toàn bộ **Chapter Timeline** được đọc tự động từ metadata nhúng bên trong file `.m4b`.
   - File `.vtt` sẽ được Audiobookshelf nhận diện làm phụ đề đồng bộ theo từng câu văn.
