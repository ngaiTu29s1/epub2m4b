# Plan: Clean Migration to VieNeu-TTS v3

Spec: `docs/superpowers/specs/2026-10-31-vieneu-tts-clean-migration-design.md`

## Phase 0 — Recon & env (DONE)
- [x] Read spec
- [x] Install `vieneu==3.8.3` (drags only kaldi-native-fbank, sea-g2p); torch/torchaudio/gradio untouched
- [x] Install `neucodec` (needed by `standard` mode for NNN fine-tune); torch/gradio untouched
- [x] Probed API: `Vieneu(mode='v3turbo')`, `list_preset_voices()`, `get_preset_voice()`, `infer(text=, voice=)`, `infer(text=, ref_audio=)`, `save(wav, path)`
- [x] 25 preset voices confirmed; default `Hải Đăng`
- [x] NNN fine-tune repo is the **v2/standard** format (`voices.json` with `codes`+`text`)

## Phase 1 — Clean up old architectures
- [ ] Delete: `core/rvc_engine.py`, `core/chatterbox_engine.py`, `vendor/viterbox_src/`,
      `models/chatterbox_vui_ve/`, `models/rvc_vui_ve/`,
      `scripts/render_chapter_chatterbox.py`, `scripts/verify_chatterbox.py`,
      `scripts/check_tail_artifacts.py`, `scripts/check_pause_alignment.py`,
      `tests/test_rvc_engine.py`, `tests/test_chatterbox_engine.py`
- [ ] Remove `F5TTSEngine`, RVC from `core/tts_engine.py`; remove Viterbox re-exports
- [ ] Decide fate of `scripts/render_chapter_vui_ve.py`, `scripts/verify_chapter_vui_ve.py`,
      `scripts/verify_chapter_dua_tre_hu.py` (old F5/RVC scripts → delete)

## Phase 2 — `core/vieneu_engine.py`
- [ ] `VieNeuEngine(TTSEngine)` with 3 modes:
  - [ ] `preset` (default) — `Vieneu(mode='v3turbo')`, `infer(voice=...)`
  - [ ] `classic` (NNN) — `Vieneu(mode='standard', backbone_repo='nguyen-brat/...')` or clone from golden ref
  - [ ] `clone` — `infer(ref_audio=..., )` for Vui Vẻ / custom WAV
- [ ] Output mono 16-bit PCM WAV at engine's native SR (48k) via ffmpeg resample to SAMPLE_RATE
- [ ] `list_available_voices()` helper for the UI
- [ ] Lazy singleton model cache so the UI doesn't reload per segment

## Phase 3 — Integrate into `core/tts_engine.py` + `app.py`
- [ ] engine registry / factory in `app.py` defaulting to VieNeu-TTS
- [ ] Dropdown: all presets + `Nguyễn Ngọc Ngạn` + `Vui Vẻ (Clone)` + `Custom Clone`
- [ ] Remove RVC `.pth/.index` inputs and F5-TTS boxes
- [ ] Keep Edge-TTS as secondary

## Phase 4 — `core/text_normalizer.py`
- [ ] `number_to_words_vi` (Bắc chuẩn: nghìn/linh/mốt/tư/lăm)
- [ ] Roman numerals → words (`Chương IV` → `Chương bốn`, `thế kỷ XXI` → `hai mươi mốt`)
- [ ] Dates → words, no `Ngày ngày`, `tháng 4` → `tháng tư`, `thứ 4` → `thứ tư`
- [ ] Units: `1m76`, `60cm`, `25%`, `1,5`
- [ ] `max_chars=500` default in `markdown_to_segments`

## Phase 5 — QA/QC with Whisper medium
- [ ] `scripts/verify_vieneu.py`
- [ ] sample WAV + WER on Whisper `medium`
- [ ] PHẦN DẪN chapter render + m4b + WER < 10%

## Phase 6 — Tests
- [ ] `tests/test_vieneu_engine.py`
- [ ] update `tests/test_text_normalizer.py`, `tests/test_tts_engine.py`, `tests/test_app_ui.py` (rename `test_app_smoke.py`?)
- [ ] `.venv/bin/pytest -q` 100% pass
