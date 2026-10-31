"""Dynamic voice catalogue + adaptive batch size (spec 2026-10-31).

Không nạp model thật: dùng fake ``tts`` với ``list_preset_voices()``.
"""
import pytest

from core import vieneu_engine
from core.vieneu_engine import (
    BATCH_SIZE_ENV,
    VOICE_CATEGORY_NEWS,
    VOICE_CATEGORY_STARS,
    VOICE_CATEGORY_STORY,
    VieNeuEngine,
    _classify_voice_label,
    auto_detect_batch_size,
    build_voice_catalogue,
    get_available_preset_voices,
    get_categorized_voices,
    list_available_preset_voices,
)

# Nhãn thật của VieNeu-TTS v3 Turbo (25 giọng).
REAL_LABELS = [
    '⭐ Adam bựa — Nam · Bắc · Phong cách tự nhiên',
    '⭐ Trúc Ly — Nữ · Bắc · Phong cách tự nhiên',
    '⭐ Thiện Minh — Nam · Bắc · Phong cách kể chuyện',
    '⭐ Mai Anh — Nữ · Bắc · Phong cách tin tức',
    '⭐ Hải Đăng — Nam · Bắc · Phong cách tự nhiên',
    '⭐ Thùy Dung — Nữ · Nam · Phong cách tin tức',
    '⭐ Thiền Tâm Đức — Nam · Bắc · Phong cách kể chuyện',
    '⭐ Ngọc Huyền — Nữ · Bắc · Giọng đọc tự nhiên',
    '⭐ Quang Sơn — Nam · Trung · Phong cách tự nhiên',
    '⭐ Ngọc Trân — Nữ · Trung · Phong cách tự nhiên',
    'Minh Đức — Nam · Bắc · Phong cách tin tức',
    'Phạm Tuyên — Nam · Bắc · Phong cách tự nhiên',
    'Thái Sơn — Nam · Nam · Phong cách kể chuyện',
    'Xuân Vĩnh — Nam · Bắc · Phong cách tự nhiên',
    'Thanh Bình — Nam · Bắc · Phong cách kể chuyện',
    'Ngọc Linh — Nữ · Bắc · Phong cách kể chuyện',
    'Đoan Trang — Nữ · Bắc · Phong cách tự nhiên',
    'Thục Đoan — Nữ · Nam · Phong cách kể chuyện',
    'Minh Triết — Nam · Nam · Phong cách tin tức',
    'Mỹ Duyên — Nữ · Nam · Phong cách đọc truyện',
    'Quỳnh Anh — Nữ · Bắc · Phong cách đọc truyện',
    'Đức Trí — Nam · Nam · Phong cách đọc truyện',
    'Kim Thanh — Nữ · Nam · Phong cách đọc truyện',
    'Adam — Nam · Nam · Giọng đọc tự nhiên',
    'Quốc Tuấn — Nam · Bắc · Phong cách tự nhiên',
]


def _name_of(label: str) -> str:
    return label.split('—')[0].replace('⭐', '').strip()


class FakeTTS:
    def __init__(self, labels=None):
        self.labels = list(labels if labels is not None else REAL_LABELS)

    def list_preset_voices(self):
        return [(label, _name_of(label)) for label in self.labels]


@pytest.fixture
def fake_tts():
    return FakeTTS()


# ---------------------------------------------------------------------------
# 2.2 Dynamic voice catalogue
# ---------------------------------------------------------------------------
def test_get_available_preset_voices_returns_all_25(fake_tts):
    voices = get_available_preset_voices(fake_tts)
    assert len(voices) == 25
    assert all(isinstance(v, tuple) and len(v) == 2 for v in voices)
    assert ('⭐ Hải Đăng — Nam · Bắc · Phong cách tự nhiên', 'Hải Đăng') in voices


def test_no_hardcoded_fallback_list_remains():
    assert not hasattr(vieneu_engine, 'FALLBACK_PRESET_VOICES')
    assert not hasattr(vieneu_engine, 'DEFAULT_BATCH_SIZE_CUDA')


def test_list_available_preset_voices_returns_names(fake_tts):
    names = list_available_preset_voices(fake_tts)
    assert len(names) == 25
    assert 'Phạm Tuyên' in names
    assert names[0] == 'Adam bựa'


def test_get_available_preset_voices_uses_live_model(monkeypatch):
    """Không truyền tts → gọi ``load_vieneu_model`` (động, không danh sách tĩnh)."""
    called = {}

    def fake_load(*_a, **_kw):
        called['yes'] = True
        return FakeTTS()

    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', fake_load)
    assert len(get_available_preset_voices()) == 25
    assert called.get('yes')


def test_get_available_preset_voices_offline_is_empty(monkeypatch):
    def boom(*_a, **_kw):
        raise RuntimeError('offline')

    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', boom)
    assert get_available_preset_voices() == []


def test_category_classification():
    assert _classify_voice_label('⭐ Adam bựa — Phong cách tự nhiên') == VOICE_CATEGORY_STARS
    assert _classify_voice_label('Ngọc Linh — Phong cách kể chuyện') == VOICE_CATEGORY_STORY
    assert _classify_voice_label('Mỹ Duyên — Phong cách đọc truyện') == VOICE_CATEGORY_STORY
    assert _classify_voice_label('Mai Anh — Phong cách tin tức') == VOICE_CATEGORY_NEWS
    assert _classify_voice_label('Quang Sơn — Phong cách tự nhiên') == VOICE_CATEGORY_NEWS
    # Nhãn lạ vẫn phải thuộc một nhóm (không rơi khỏi catalogue).
    assert _classify_voice_label('Giọng lạ') == VOICE_CATEGORY_NEWS


def test_get_categorized_voices_buckets(fake_tts):
    cats = get_categorized_voices(fake_tts)
    assert set(cats) == {VOICE_CATEGORY_STARS, VOICE_CATEGORY_STORY, VOICE_CATEGORY_NEWS}
    assert len(cats[VOICE_CATEGORY_STARS]) == 10
    # 25 giọng phải được phân bổ hết, không mất giọng nào.
    total = sum(len(v) for v in cats.values())
    assert total == 25
    names = {name for bucket in cats.values() for _label, name in bucket}
    assert 'Phạm Tuyên' in names and 'Adam bựa' in names


def test_get_categorized_voices_story_and_news_non_empty(fake_tts):
    cats = get_categorized_voices(fake_tts)
    assert cats[VOICE_CATEGORY_STORY], 'phải có giọng kể chuyện'
    assert cats[VOICE_CATEGORY_NEWS], 'phải có giọng tin tức/tự nhiên'


def test_categorized_voice_labels_carry_metadata(fake_tts):
    cats = get_categorized_voices(fake_tts)
    story = cats[VOICE_CATEGORY_STORY]
    assert any('kể chuyện' in label or 'đọc truyện' in label for label, _n in story)


def test_build_voice_catalogue_uses_live_presets(monkeypatch):
    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', lambda *a, **k: FakeTTS())
    options = build_voice_catalogue()
    presets = [o for o in options if o.mode == 'preset']
    assert len(presets) == 25
    assert all(o.voice for o in presets)
    assert any(o.voice == 'Phạm Tuyên' for o in presets)
    # Không còn nhãn "Giọng sẵn có" cũ.
    assert all('Giọng sẵn có' not in o.label for o in presets)


def test_build_voice_catalogue_keeps_classic_and_clones(monkeypatch):
    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', lambda *a, **k: FakeTTS())
    labels = [o.label for o in build_voice_catalogue()]
    assert any('Nguyễn Ngọc Ngạn' in x for x in labels)
    assert any('Clone · Vui Vẻ' in x for x in labels)
    assert any('Clone · WAV' in x for x in labels)


def test_build_voice_catalogue_groups_by_category(monkeypatch):
    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', lambda *a, **k: FakeTTS())
    labels = [o.label for o in build_voice_catalogue() if o.mode == 'preset']
    assert any(x.startswith('Giọng tuyển chọn') for x in labels)
    assert any(x.startswith('Giọng kể chuyện') for x in labels)
    assert any(x.startswith('Giọng tin tức') for x in labels)


# ---------------------------------------------------------------------------
# 2.1 Adaptive batch size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'vram,expected',
    [
        (None, 1),      # CPU
        (4.0, 2),       # GPU < 6GB
        (8.0, 4),       # 6–10GB
        (12.0, 8),      # RTX 3060 12GB — điểm ngọt
        (16.0, 16),     # ≥ 16GB
        (24.0, 16),
    ],
)
def test_batch_size_table(vram, expected):
    assert vieneu_engine._batch_size_for_vram(vram) == expected


def test_auto_detect_prefers_env(monkeypatch):
    monkeypatch.setenv(BATCH_SIZE_ENV, '12')
    monkeypatch.setattr(vieneu_engine, '_vram_gb', lambda: 12.0)
    assert auto_detect_batch_size() == 12


def test_auto_detect_env_clamps_to_one(monkeypatch):
    monkeypatch.setenv(BATCH_SIZE_ENV, '0')
    monkeypatch.setattr(vieneu_engine, '_vram_gb', lambda: 12.0)
    assert auto_detect_batch_size() == 8  # giá trị không hợp lệ → bảng VRAM


def test_auto_detect_ignores_garbage_env(monkeypatch):
    monkeypatch.setenv(BATCH_SIZE_ENV, 'abc')
    monkeypatch.setattr(vieneu_engine, '_vram_gb', lambda: 12.0)
    assert auto_detect_batch_size() == 8


def test_auto_detect_without_env_uses_vram(monkeypatch):
    monkeypatch.delenv(BATCH_SIZE_ENV, raising=False)
    monkeypatch.setattr(vieneu_engine, '_vram_gb', lambda: 12.0)
    assert auto_detect_batch_size() == 8


def test_auto_detect_on_cpu(monkeypatch):
    monkeypatch.delenv(BATCH_SIZE_ENV, raising=False)
    monkeypatch.setattr(vieneu_engine, '_vram_gb', lambda: None)
    assert auto_detect_batch_size() == 1


def test_engine_uses_auto_detect(monkeypatch):
    monkeypatch.setenv(BATCH_SIZE_ENV, '6')
    engine = VieNeuEngine(device='cuda')
    assert engine.batch_size == 6


def test_engine_explicit_batch_size_wins(monkeypatch):
    monkeypatch.setenv(BATCH_SIZE_ENV, '6')
    engine = VieNeuEngine(device='cuda', batch_size=3)
    assert engine.batch_size == 3
