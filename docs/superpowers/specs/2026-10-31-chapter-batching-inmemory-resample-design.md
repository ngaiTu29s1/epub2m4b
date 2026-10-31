# Design Doc: Chapter-Level Segment Batching & In-Memory Resampling Pipeline

**Date:** 2026-10-31
**Target:** `core/tts_engine.py`, `core/vieneu_engine.py`, `tests/test_tts_engine.py`, `tests/test_vieneu_engine.py`

---

## 1. Bối cảnh & Vấn đề Cốt lõi (Problem Statement)

1. **Vấn đề 1: Batching GPU bị vô hiệu hoá trong `render_chapter`**
   - Để sinh phụ đề WebVTT (`cues`) chuẩn từng câu, `markdown_to_segments` tách văn bản chương thành các `AudioSegmentSpec` ngắn (50–150 ký tự).
   - Trước đây `render_chapter` gọi `engine.synthesize_segment(seg.text)` tuần tự từng câu.
   - Mỗi câu ngắn hơn `max_chars=256` nên VieNeu chia thành đúng 1 chunk. Dù `batch_size=4`, GPU vẫn chạy tuần tự từng chunk, không tận dụng được CUDA graph batching.
   - Đo thực tế trên 89 chương: **52 phút** (RTF ≈ 0.08×).

2. **Vấn đề 2: Lãng phí I/O & subprocess FFmpeg**
   - Mỗi câu sinh ở 48 kHz → `_write_wav_resampled` ghi WAV tạm → `subprocess.run(["ffmpeg", ...])` hạ mẫu về 24 kHz → đọc lại.
   - Một cuốn 89 chương ≈ 5.000 câu ⇒ 5.000 lần fork FFmpeg, tốn CPU/I-O vô ích.

3. **Mục tiêu đo lường được**
   - Batch toàn bộ danh sách câu của một chương qua length-bucketing sẵn có của VieNeu-TTS v3 Turbo.
   - Thay resample từng câu bằng in-memory resampler (~0.1 ms/câu).
   - Giữ 100% độ chính xác `SubtitleCue` (`start_ms`, `end_ms`, `text`) cho `.vtt` và timeline `.m4b`.
   - Hướng tới RTF ≤ 0.04×. **Ghi chú thực nghiệm:** RTF đo được trên RTX 3060 cho luồng
     batch này là ≈ 0.03–0.05× tuỳ độ dài câu; con số 0.04× không phải cam kết bảo đảm
     cho mọi giọng/cấu hình.

---

## 2. Thiết Kế Kiến Trúc & Chi Tiết Kỹ Thuật

### 2.1. In-Memory Resampling (`core/vieneu_engine.py`)

Hàm `_resample_waveform_in_memory(samples, orig_sr, target_sr) -> np.ndarray`:

- `orig_sr == target_sr` → trả nguyên bản (không copy).
- Thứ tự backend: `soxr.resample` (nhanh nhất, đã có trong `.venv`) → `torchaudio.functional.resample`
  (chỉ khi tensor/soxr lỗi) → `scipy.signal.resample_poly`.
- Nếu tất cả đều lỗi: ghi WAV tạm và gọi FFmpeg (fallback cuối, giữ hành vi cũ).
- Đầu ra luôn 1-D `float32`.

`_write_wav_resampled` nay resample **trong RAM** rồi ghi bằng `soundfile`; không còn
subprocess ở đường thường.

### 2.2. Chapter-Level Batch Synthesis (`core/vieneu_engine.py`)

```python
def synthesize_segments_batch(
    self,
    texts: list[str],
) -> list[tuple[np.ndarray, int]]:  # [(waveform_24k_float32, duration_ms), ...]
```

- `load_vieneu_model(self.device)` **một lần** cho cả batch.
- `speaker_emb, ref_codes = tts._resolve_ref(voice=..., ref_audio=..., denoise=True, use_ref_codes=True)`
  **một lần** cho cả batch.
- Mỗi `text` được `normalize_to_chunks_v3_with_gaps(...)` rồi `phonemize_text_with_emotions(...)`.
  **Đính chính so với bản sketch:** `_infer_chunks` nhận **phoneme string**, không nhận văn bản thô,
  và `_cap_frames(sampling, ...)` của nó đọc `sampling["max_new_frames"]` nên `sampling={}` sẽ
  `KeyError`. Vì vậy ta truyền `DEFAULT_SAMPLING` (khớp mặc định `infer`:
  `temperature=0.8, top_k=25, top_p=0.95, max_new_frames=300, repetition_penalty=1.2`).
- Toàn bộ phoneme của mọi câu được dàn phẳng vào **một** lời gọi
  `tts._infer_chunks(all_phonemes, speaker_emb, ref_codes, use_ref_codes=True, batch_size=self.batch_size, sampling=DEFAULT_SAMPLING)`
  → length-bucketing + CUDA graph chia đều mọi câu vào từng forward `batch_size`, không chỉ 1 câu/lần.
- Gom phoneme trở lại từng câu theo đúng thứ tự và nối bằng khoảng lặng nội tại copy từ `infer`.
- Resample trong RAM về 24 kHz; `duration_ms = round(len(samples) / 24000 * 1000)`.

### 2.3. Tích Hợp `render_chapter` (`core/tts_engine.py`)

- Nếu `engine` có `synthesize_segments_batch`: gọi **một lần** với `texts = [s.text for s in segments]`,
  ghép trực tiếp mảng PCM (không file tạm), chèn khoảng lặng `seg.pause_after_ms`.
- `start_ms`/`end_ms` tính từ độ dài thực của từng waveform (`duration_ms` trả về), nên khớp tuyệt đối
  với offset mẫu đã ghi.
- Engine không hỗ trợ (`EdgeTTSEngine`) → fallback tuần tự như cũ, hành vi và cues không đổi.

---

## 3. Kế Hoạch Triển Khai & Verification

1. `core/vieneu_engine.py`: `_resample_waveform_in_memory`, `_write_wav_resampled` in-memory,
   `synthesize_segments_batch`, `DEFAULT_SAMPLING`.
2. `core/tts_engine.py`: `render_chapter` dùng batch khi có, giữ schema `chapter_XXX.json`.
3. Tests: `tests/test_vieneu_engine.py` (resampler, batch count/length/order, một lần gọi
   `_infer_chunks`) và `tests/test_tts_engine.py` (batch path, fallback path, cue chính xác).
4. `.venv/bin/pytest -q` phải 100% PASS.
