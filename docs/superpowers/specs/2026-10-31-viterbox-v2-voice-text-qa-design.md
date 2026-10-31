# PRD & Spec: Viterbox v2 — Multi-Clip Voice Conditioning, Context-Aware Vietnamese Normalizer & Prosody Upgrade

**Date:** 2026-10-31  
**Project:** `epub2m4b`  
**Target Branch:** `feat/mvp`  
**OpenCode Session:** `ses_e6fad7311ffeeDj17ehf5DWyps` (`omniroute/deepseek-v4-flash`)

---

## 1. Problem Statement & User Feedback

Qua quá trình nghe thử giọng mẫu và tạo thử `chapter_001.wav` trên giao diện Studio, người dùng đánh giá chất lượng hiện tại đạt khoảng **7/10 điểm** và chỉ ra 5 vấn đề cốt lõi cần giải quyết triệt để:

1. **Đọc sai số La Mã, ngày tháng năm, chữ số trong câu mẫu và sách:**
   - Câu test trên UI: `"Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi."`
   - Nguyên nhân gốc rễ:
     - `normalize_vi_text()` trong `core/text_normalizer.py` chỉ biến `"Chương IV: Ngày 19/05/1990"` thành `"Chương 4: Ngày 19 tháng 5 năm 1990"` (vẫn giữ nguyên chữ số `4`, `19`, `5`, `1990`).
     - Bên trong `vendor/viterbox_src/viterbox/tts.py`, thư viện `vinorm` không tồn tại (`HAS_VINORM = False`), nên chữ số Ả Rập đi thẳng vào bộ character tokenizer của `T3`.
     - Khi `T3` gặp chữ số (`4`, `19`, `5`, `1990`), nó đọc loạn thành *"Trường có, ngay tràn kháng năm mơ trăm ờ..."*. Tương tự, `"CHƯƠNG 1"` trong Chương 1 cũng còn nguyên số `1`.
     - Ngoài ra, số La Mã đứng ở các vị trí khác (như `"thế kỷ XXI"`, `"I."`, `"II."`, `"Vua Lê Thánh Tông II"`) hoặc các từ tiếng Anh / tên quốc gia / đơn vị đo lường (`1m76`, `60cm`, `km/h`, `25%`, `1,5`) chưa được chuyển thành chữ tiếng Việt theo đúng ngữ cảnh.

2. **Lãng phí kho 226 clips (27 phút) đã cào & Giọng đôi khi bị trôi sang accent miền Nam:**
   - Hiện tại `ViterboxTTSEngine` chỉ lấy đúng **1 file `golden_01.wav` (9 giây)** để làm one-shot prompt, bỏ phí hoàn toàn 225 clips sạch còn lại trong `voices/vui_ve/wavs/`.
   - Tệ hơn, `golden_01.wav` lại chứa câu có nhiều từ tiếng Anh viết tắt (`"SCP"`, `"SCP-610"`, `"robot"`, `"data"`)!
   - Vì chỉ lấy embedding từ 1 clip lẫn tiếng Anh + `cfg_weight=0.35` hơi thấp, mô hình `T3` (vốn pre-train trên 3.000 giờ sách nói Bắc – Trung – Nam) đôi khi bị trôi về giọng miền Nam ở một số từ và không khóa chặt được **accent thuần Bắc** đặc trưng của kênh Vui Vẻ.

3. **Ngắt nghỉ trong Chapter 1 còn cứng ngắc:**
   - Có **2 tầng cắt câu chồng chéo nhau**:
     - Tầng 1: `markdown_to_segments()` trong `core/text_normalizer.py` cắt các câu dài $> 280$ ký tự tại dấu phẩy thành nhiều `AudioSegmentSpec` rời rạc và chèn **400ms dead silence** ở giữa câu! (Ví dụ câu 2 của Chương 1 bị cắt đôi sau chữ *"chắc chắn,"* bằng 400ms im lặng chết).
     - Tầng 2: `plan_viterbox_segments()` trong `core/chatterbox_engine.py` lại cắt câu lần nữa và chèn 200ms dead silence (`np.zeros`) không có crossfade hay room tone mượt.

4. **Đôi khi bị nuốt chữ (đặc biệt âm cuối trước dấu câu):**
   - Trong `vendor/viterbox_src/viterbox/tts.py`, việc cắt cứng 2 speech tokens cuối (`speech_tokens[:-2]`) cộng với `tail_cleanup_clean(valley_rms=0.0055, valley_frames=3)` đôi khi cắt lẹm vào phụ âm cuối hoặc nguyên âm thanh huyền/nặng ở cuối mệnh đề.

5. **QA/QC cần dùng Whisper chuẩn hơn thay vì `whisper small`:**
   - `faster-whisper` bản `small` thường xuyên nghe nhầm thanh điệu tiếng Việt miền Bắc (khiến báo cáo WER bị nhiễu sai). Cần nâng cấp bộ QA/QC (`scripts/verify_chatterbox.py` và pipeline kiểm thử) lên dùng **`medium`** (hoặc `large-v3` / `large-v3-turbo` nếu có sẵn) làm thước đo chính thức, đồng thời kiểm tra cả câu mẫu có số/ngày tháng/La Mã lẫn Chương 1.

---

## 2. Goals & Non-Goals

### Goals
1. **Context-Aware Pure-Python Vietnamese Text Normalizer (Thuần Bắc 100%):**
   - Mọi chữ số, số La Mã, ngày tháng năm, giờ giấc, số thập phân, phần trăm, đơn vị đo lường, chữ viết tắt, tên quốc gia/thành phố phổ biến và từ tiếng Anh thông dụng đều được chuyển thành **chữ tiếng Việt thuần Bắc** (`"một nghìn"`, `"linh"`, `"tư"`, `"lăm"`, `"mốt"`) trước khi vào `T3` — **0% còn sót chữ số 0-9 trong chuỗi đưa vào tokenizer của T3**.
2. **Tận dụng toàn bộ kho Voice Harvester (Multi-Clip Speaker Conditioning + Pure-Vietnamese Reference Pool):**
   - Không còn "one-shot 1 file". Tự động nạp **Top 30–50 clips sạch nhất, thuần Việt nhất** từ `voices/vui_ve/profile.json` để tính **Ensemble Multi-Clip Speaker Embedding (`ve_embed`)**, kết hợp chọn **Golden Reference 100% thuần tiếng Việt miền Bắc** (không có từ tiếng Anh `SCP`/`robot`/`data`) cho `S3Gen` và `T3` prompt tokens.
   - Tinh chỉnh `cfg_weight` và `temperature` để khóa chặt **accent thuần Bắc** của Vui Vẻ, không trôi sang giọng miền Nam.
3. **Nhịp đọc (Prosody) liền mạch & Không nuốt chữ:**
   - Loại bỏ cắt câu giữa chừng ở tầng `markdown_to_segments` (chỉ tách ở ranh giới kết thúc câu `.!?…`, không bao giờ cắt câu ở dấu phẩy thành 2 segment có dead silence 400ms).
   - Giữ trọn âm tiết cuối câu (không nuốt chữ) trong khi vẫn đảm bảo **0% tiếng rè ở đoạn ngắt**.
4. **QA/QC nghiêm ngặt với Whisper `medium`:**
   - Nâng cấp `scripts/verify_chatterbox.py` bỏ hẳn sự phụ thuộc vào `whisper small`, dùng `medium` làm chuẩn duy nhất và bổ sung bài test QA cho cả:
     1. Câu mẫu `"Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi."` (WER = 0%).
     2. Chương `# PHẦN DẪN` (`output_viterbox_tts.wav`).
     3. Đoạn trích Chương 1 (`CHƯƠNG 1`).

### Non-Goals
- Không tạo thêm bất kỳ virtualenv nào ngoài `.venv` hiện tại.
- Không quay lại F5-TTS hay RVC.

---

## 3. Detailed Technical Design

### 3.1. Module 1: Bộ chuẩn hoá văn bản tiếng Việt đa ngữ cảnh (Pure Northern Vietnamese Normalizer)
Cập nhật `core/text_normalizer.py` và tích hợp vào `normalize_viterbox_text()` (`core/chatterbox_engine.py`) + `normalize_text()` (`vendor/viterbox_src/viterbox/tts.py`):

1. **Chuyển đổi Số nguyên & Số thứ tự chuẩn miền Bắc (`number_to_words_vi`):**
   - Hỗ trợ từ `0` đến hàng tỷ (`tỷ`, `triệu`, `nghìn`, `trăm`).
   - Áp dụng đúng quy tắc ngữ âm tiếng Việt miền Bắc:
     - `1` ở hàng đơn vị khi hàng chục $\ge 2$ $\to$ `"mốt"` (`21` $\to$ `"hai mươi mốt"`).
     - `4` ở hàng đơn vị khi hàng chục $\ge 2$ $\to$ `"tư"` (`24` $\to$ `"hai mươi tư"`).
     - `5` ở hàng đơn vị khi hàng chục $\ge 1$ $\to$ `"lăm"` (`15` $\to$ `"mười lăm"`, `25` $\to$ `"hai mươi lăm"`).
     - Hàng chục bằng `0` và hàng đơn vị $\ne 0$ $\to$ dùng **`"linh"`** (thuần Bắc, không dùng `"lẻ"` miền Nam): `105` $\to$ `"một trăm linh năm"`, `2004` $\to$ `"hai nghìn không trăm linh tư"`.
     - Dùng **`"nghìn"`** (thuần Bắc, không dùng `"ngàn"`): `1990` $\to$ `"một nghìn chín trăm chín mươi"`.
2. **Xử lý theo Ngữ cảnh (Context-Aware Rules):**
   - **Ngày tháng năm:**
     - `dd/mm/yyyy` hoặc `dd-mm-yyyy`: `"19/05/1990"` $\to$ `"mười chín tháng năm năm một nghìn chín trăm chín mươi"`. (Nếu trước đó đã có chữ `"Ngày"` hoặc `"ngày"`, không lặp lại chữ `"ngày"` thành `"Ngày ngày mười chín"`!).
     - `"tháng 4"` $\to$ `"tháng tư"` (riêng tháng 4 trong tiếng Việt đọc là `"tháng tư"`, `"thứ 4"` đọc là `"thứ tư"`).
     - `dd/mm`: `"30/4"` $\to$ `"ba mươi tháng tư"`, `"1/5"` $\to$ `"mùng một tháng năm"` (khi đứng sau `"ngày"` và $d \le 10$ hoặc đọc `"một tháng năm"`).
     - `mm/yyyy`: `"05/1990"` $\to$ `"tháng năm năm một nghìn chín trăm chín mươi"`.
   - **Số La Mã theo ngữ cảnh (`I` – `XXX`):**
     - Sau các từ chỉ mục/thứ tự: `Chương`, `Phần`, `Quyển`, `Tập`, `Hồi`, `Mục`, `Thế kỷ`, `Thế chiến`, `Lần`, `Cấp`, `Loại`, `Vành đai`, `Quý` $\to$ chuyển sang chữ số tiếng Việt (`"Chương IV"` $\to$ `"Chương bốn"`, `"thế kỷ XXI"` $\to$ `"thế kỷ hai mươi mốt"`).
     - Đầu dòng làm đề mục: `"I. Mở đầu"`, `"II. Nội dung"`, `"IV: ..."` $\to$ `"Một. Mở đầu"`, `"Hai. Nội dung"`, `"Bốn, ..."`.
     - Sau tên riêng vua/nữ hoàng/giáo hoàng (ví dụ `"Elizabeth II"`) $\to$ `"đệ nhị"` hoặc `"thứ hai"`.
   - **Đơn vị đo lường, Giờ giấc, Số thập phân, Phần trăm, Khoảng số:**
     - `1m76` $\to$ `"một mét bảy mươi sáu"`, `60cm` $\to$ `"sáu mươi xen ti mét"`, `30km/h` $\to$ `"ba mươi ki lô mét trên giờ"`, `50kg` $\to$ `"năm mươi ki lô gam"`, `24kHz` $\to$ `"hai mươi tư ki lô héc"`.
     - `14h30` / `14:30` $\to$ `"mười bốn giờ ba mươi phút"`.
     - `1,5` / `3.14` $\to$ `"một phẩy năm"`, `"ba chấm mười bốn"`.
     - `25%` $\to$ `"hai mươi lăm phần trăm"`.
     - Khoảng số: `6-7 mét` / `10 - 15 phút` $\to$ `"sáu đến bảy mét"`, `"mười đến mười lăm phút"`.
     - Mã số có số 0 đứng đầu (`007`) $\to$ đọc từng số `"không không bảy"`.
   - **Từ viết tắt, Tên quốc gia & Từ tiếng Anh thông dụng:**
     - Mở rộng `_ABBR_MAP`: `TP.HCM` / `TPHCM` $\to$ `"Thành phố Hồ Chí Minh"`, `HN` $\to$ `"Hà Nội"`, `TW` $\to$ `"Trung ương"`, `UBND` $\to$ `"Ủy ban nhân dân"`, `ĐHQG` $\to$ `"Đại học Quốc gia"`, `THPT` $\to$ `"Trung học phổ thông"`, `THCS` $\to$ `"Trung học cơ sở"`, `NXB` $\to$ `"Nhà xuất bản"`, `GS` $\to$ `"Giáo sư"`, `PGS` $\to$ `"Phó giáo sư"`, `TS` $\to$ `"Tiến sĩ"`, `BS` $\to$ `"Bác sĩ"`, `KS` $\to$ `"Kỹ sư"`.
     - Từ tiếng Anh/nước ngoài phổ biến trong sách & truyện: ánh xạ sang âm đọc tiếng Việt tự nhiên (ví dụ `Internet` $\to$ `"in tơ nét"`, `camera` $\to$ `"ca mê ra"`, `robot` $\to$ `"rô bốt"`, `logic` $\to$ `"lô gíc"`, `video` $\to$ `"vi đê ô"`, `radio` $\to$ `"ra đi ô"`, `TV` $\to$ `"ti vi"`, `ok` / `OK` $\to$ `"ô kê"`, `livestream` $\to$ `"lai chim"`, `online` $\to$ `"on lai"`, `facebook` $\to$ `"phây búc"`, `youtube` $\to$ `"diu túp"`, `New York` $\to$ `"Niu Óoc"`, `Tokyo` $\to$ `"Tô ki ô"`, `Paris` $\to$ `"Pa ri"`, `London` $\to$ `"Luân Đôn"`, `Washington` $\to$ `"Oa sinh tơn"`).
   - **Chốt chặn cuối cùng (Fallback Digit Sweep):**
     - Bất kỳ chuỗi số `\d+` nào còn sót lại sau tất cả các bước trên đều bắt buộc đi qua `number_to_words_vi()` để đảm bảo **không bao giờ còn ký tự `0-9` đi vào `T3`**.

### 3.2. Module 2: Multi-Clip Voice Conditioning & Khóa Accent Thuần Bắc (`core/chatterbox_engine.py` & `vendor/viterbox_src/viterbox/tts.py`)
1. **Tuyển chọn bộ Reference Thuần Việt từ 226 clips (`voices/vui_ve/profile.json`):**
   - Viết hàm `select_pure_vietnamese_refs(voice_dir: Path, top_k_embed: int = 40)` trong `core/chatterbox_engine.py`:
     - Đọc `profile.json`, lọc ra các clips **100% thuần tiếng Việt** (không chứa chữ số, không chứa từ tiếng Anh như `SCP`, `Love`, `Luffy`, `Euclid`, `Keter`, `Marvel`, không chứa ký tự `*`), có `quality_score >= 0.85` và độ dài `6.0s – 9.5s`.
     - Chọn clip thuần Việt có chất lượng & nhịp điệu chuẩn nhất (ví dụ `clip_0229.wav` hoặc `clip_0104.wav` — 100% tiếng Việt miền Bắc, phát âm cực tròn vành rõ chữ) làm **Primary Acoustic Prompt** cho `S3Gen` và `T3` speech prompt tokens (hoặc xuất ra `voices/vui_ve/golden_refs/golden_pure_vi.wav`).
     - Lấy **Top 35–40 clips thuần Việt** nạp vào `self.ve.embeds_from_wavs(wav_list, sample_rate=16000)` rồi lấy trung bình cộng chuẩn hoá L2 (`mean(axis=0)`) làm **Ensemble Speaker Embedding (`ve_embed`)**!
2. **Hỗ trợ `extra_ref_wavs` trong `prepare_conditionals` (`vendor/viterbox_src/viterbox/tts.py`):**
   - Cho phép truyền danh sách `speaker_embed_wavs: Optional[List[Union[str, Path]]] = None` vào `prepare_conditionals()`.
   - Khi có `speaker_embed_wavs`, tính `ve_embed` trung bình trên toàn bộ 35–40 files WAV đã cào thay vì chỉ 1 file lẻ. Điều này tận dụng triệt để công sức cào 27 phút voice và khóa cứng đặc trưng thanh quản + accent miền Bắc của Vui Vẻ.
3. **Tinh chỉnh Hyperparameters khóa Accent Bắc:**
   - Tăng `cfg_weight` từ `0.35` lên `0.50` (giúp `T3` bám chặt vào conditioning miền Bắc của Vui Vẻ thay vì trôi về prior miền Nam của tập pre-train).
   - Giữ `temperature = 0.65`, `exaggeration = 0.42`, `prompt_time_stretch = 0.90`.

### 3.3. Module 3: Sửa Ngắt Nghỉ Cứng Ngắc & Chống Nuốt Chữ Cuối
1. **Bỏ cắt câu ở dấu phẩy trong `markdown_to_segments` (`core/text_normalizer.py`):**
   - Nâng `max_chars` mặc định của `markdown_to_segments` từ `280` lên `600` (hoặc chỉ tách câu tại dấu kết câu `.!?…`). Tuyệt đối không để `markdown_to_segments` tự tiện cắt một câu văn ở dấu phẩy thành 2 `AudioSegmentSpec` có `pause_after_ms = 400ms`! Việc chia mệnh đề bên trong một câu dài do `ViterboxTTSEngine` toàn quyền kiểm soát.
2. **Nối mệnh đề mượt mà trong `ViterboxTTSEngine.synthesize_segment` (`core/chatterbox_engine.py`):**
   - Khi một câu dài ($\ge 32$ từ) bắt buộc phải tách thành 2 mệnh đề tại dấu phẩy:
     - Thay vì chèn 200ms dead-zero cứng ngắc, sử dụng khoảng nghỉ ngắn tự nhiên `130ms` kèm fade mượt.
     - Giữa các câu hoàn chỉnh trong cùng đoạn văn (`pause_after_ms` trong `markdown_to_segments`), giảm từ `400ms` xuống `320ms` (cuối đoạn văn `650ms`) để nhịp đọc liền mạch, cuốn hút hơn.
3. **Sửa lỗi nuốt chữ cuối câu (`vendor/viterbox_src/viterbox/tts.py`):**
   - Trong `_generate_single`: chỉ cắt **1 token cuối** (`speech_tokens = speech_tokens[:-1]` khi `len > 2`) thay vì cắt 2 tokens (`[:-2]`), vì cắt 2 tokens tương đương mất ~80ms âm thanh cuối và làm cụt phụ âm cuối câu!
   - Trong `tail_cleanup_clean`:
     - Nới `valley_frames` từ `3` (30ms) lên `5` (50ms) và `valley_rms = 0.0045`, đồng thời sau khi tìm thấy `valley_start`, **giữ lại 25ms đệm (margin = 2–3 frames) đầu thung lũng** trước khi fade-out 25ms!
     - Nhờ giữ 25ms đệm ngay sau chữ cuối cùng (trước khi cụm rè ở cuối thung lũng xuất hiện), âm tiết cuối câu có đủ không gian ngân tự nhiên 100% mà cụm nhiễu rè ở phía sau thung lũng vẫn bị cắt bỏ hoàn toàn!

### 3.4. Module 4: Nâng cấp QA/QC với Whisper `medium`
- Cập nhật `scripts/verify_chatterbox.py`:
  - Sử dụng `WhisperModel('medium')` làm bộ đánh giá ASR chính thức (bỏ `small` khỏi tiêu chuẩn chấm lỗi vì `small` nhận diện sai dấu tiếng Việt miền Bắc).
  - Thêm bài kiểm tra bắt buộc cho câu mẫu có số La Mã + ngày tháng + viết tắt:
    `"Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi."`
    $\to$ Phải đọc chuẩn 100% thành *"Chương bốn, ngày mười chín tháng năm năm một nghìn chín trăm chín mươi tại Thành phố Hồ Chí Minh, những cơn mưa đầu mùa bắt đầu rơi."*
  - Kiểm tra đồng thời `check_tail_artifacts.py` (0% đuôi rè) và `check_pause_alignment.py` (100% khoảng ngắt đúng dấu câu).
