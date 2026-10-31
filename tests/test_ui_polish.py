import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import app


def test_get_voice_showcase_sample_known_and_unknown():
    # Known voice by simple name
    sample_path, badge = app.get_voice_showcase_sample('Adam bựa')
    assert sample_path is not None, 'phải có mẫu cho "Adam bựa"'
    assert Path(sample_path).exists()
    assert Path(sample_path).suffix in ('.ogg', '.wav')
    assert 'Adam bựa' in badge

    # Known voice by full dropdown label
    sample_path2, badge2 = app.get_voice_showcase_sample(
        'Giọng tuyển chọn ⭐ · Thiện Minh — Nam · Bắc · Phong cách kể chuyện'
    )
    assert sample_path2 is not None
    assert Path(sample_path2).exists()
    assert 'Thiện Minh' in badge2

    # Unknown or empty voice
    sample_path3, badge3 = app.get_voice_showcase_sample('')
    assert sample_path3 is None
    assert badge3 == ''

    sample_path4, badge4 = app.get_voice_showcase_sample('Giọng Không Tồn Tại 123')
    assert sample_path4 is None
    assert badge4 == ''


def test_showcase_does_not_confuse_prefix_voices():
    """Regression: "Adam" là tiền tố của "Adam bựa" — không được chọn nhầm."""
    path_adam, badge_adam = app.get_voice_showcase_sample('Adam')
    path_bua, badge_bua = app.get_voice_showcase_sample('Adam bựa')
    assert path_adam and path_bua
    assert path_adam != path_bua
    assert Path(path_adam).name != Path(path_bua).name
    assert 'Adam bựa' not in badge_adam
    assert '**Adam**' in badge_adam
    assert 'Adam bựa' in badge_bua


def test_showcase_resolves_every_preset_dropdown_label(monkeypatch):
    """Mọi nhãn preset trên dropdown phải phát đúng mẫu của chính giọng đó."""
    from core import vieneu_engine
    from tests.test_voice_catalogue import REAL_LABELS, FakeTTS

    # Catalog giọng động cần model; unit test không được tải model/tới HF.
    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', lambda *a, **k: FakeTTS(REAL_LABELS))
    from core.vieneu_engine import build_voice_catalogue

    presets = [o for o in build_voice_catalogue() if o.mode == 'preset']
    assert len(presets) == 25, f'cần đủ 25 giọng, có {len(presets)}'
    for option in presets:
        sample_path, badge = app.get_voice_showcase_sample(option.label)
        assert sample_path is not None, f'không có mẫu cho {option.label!r}'
        assert Path(sample_path).exists()
        assert option.voice in badge, f'{option.voice!r} không khớp badge {badge!r}'


def test_showcase_fuzzy_ambiguous_returns_none():
    """Khớp mờ nhiều ứng viên → trả None thay vì đoán bừa."""
    assert app._fuzzy_single_match(
        [{'name': 'Adam'}, {'name': 'Adam bựa'}], 'ada'
    ) is None
    assert app._fuzzy_single_match([{'name': 'Trúc Ly'}], 'trúc') == {'name': 'Trúc Ly'}


def test_showcase_missing_index_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'DEFAULT_WORKSPACE', tmp_path)
    assert app.get_voice_showcase_sample('Adam bựa') == (None, '')


def test_extract_uncertainty_rows():
    state = {
        'chapters': [
            {
                'enabled': True,
                'markdown': 'Công thức E = mc^2 và H2SO4 cùng 1/2 cốc nước.',
            },
            {
                'enabled': False,
                'markdown': 'Nội dung bị bỏ qua: x + y = z.',
            },
        ]
    }
    rows = app.extract_uncertainty_rows(state)
    assert len(rows) >= 2
    tokens = [r[0] for r in rows]
    assert 'E = mc^2' in tokens
    assert '1/2' in tokens

    # Empty state handling
    assert app.extract_uncertainty_rows({}) == []


@patch('core.abs_client.ABSClient.check_connection')
@patch('core.abs_client.ABSClient.get_libraries')
def test_handle_test_abs_connection_success(mock_get_libs, mock_check):
    mock_check.return_value = {
        'connected': True,
        'username': 'homelab_admin',
        'server_version': '2.8.0',
    }
    mock_get_libs.return_value = [{'name': 'Sách Nói Tổng Hợp'}]

    res = app.handle_test_abs_connection('http://abs.lan:13378', 'valid-token')
    assert 'Kết nối thành công' in res
    assert 'homelab_admin' in res
    assert 'v2.8.0' in res
    assert 'Sách Nói Tổng Hợp' in res


@patch('core.abs_client.ABSClient.check_connection')
def test_handle_test_abs_connection_failure(mock_check):
    mock_check.return_value = {
        'connected': False,
        'error': 'Connection refused',
    }
    res = app.handle_test_abs_connection('http://abs.lan:13378', 'invalid-token')
    assert 'Không thể kết nối Audiobookshelf' in res
    assert 'Connection refused' in res


def test_handle_epub_upload_ui(tmp_path: Path, monkeypatch):
    epub_path = tmp_path / 'test_sample.epub'
    # Create minimal epub
    from ebooklib import epub
    book = epub.EpubBook()
    book.set_identifier('test-1')
    book.set_title('Quyển Sách Mẫu')
    c1 = epub.EpubHtml(title='Chương 1', file_name='c1.xhtml', lang='vi')
    c1.content = '<h1>Chương 1</h1><p>Vận tốc ánh sáng c = 300000 km/s và 3/4 lít dầu.</p>'
    book.add_item(c1)
    book.spine = ['nav', c1]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(epub_path), book)

    res = app.handle_epub_upload_ui(str(epub_path), work_root=tmp_path / 'work')
    assert len(res) == 7
    state, title, author, cover, choices, md_text, unc_rows = res
    assert title == 'Quyển Sách Mẫu'
    assert len(unc_rows) >= 1
    tokens = [r[0] for r in unc_rows]
    assert '3/4' in tokens


@patch('core.abs_client.ABSClient.sync_audiobook')
@patch('core.abs_client.ABSClient.is_configured', return_value=True)
def test_handle_generate_m4b_with_abs_sync(mock_is_cfg, mock_sync, tmp_path: Path, monkeypatch):
    import wave

    class DummyEngine:
        def synthesize_segment(self, text, out_wav):
            sr = 24000
            n_frames = sr // 4
            with wave.open(str(out_wav), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(b'\x00\x00' * n_frames)
            return 250

    monkeypatch.setattr(app, '_create_tts_engine', lambda *a, **kw: DummyEngine())

    monkeypatch.setattr(app, 'package_m4b', lambda meta, results, out: out)
    monkeypatch.setattr(app, 'export_webvtt', lambda results, out: out)

    mock_sync.return_value = {'success': True, 'method': 'direct_storage_copy', 'scanned': True}

    state = {'book_dir': str(tmp_path / 'book')}
    m4b_out, m4b_file, vtt_file, timeline, status = app.handle_generate_m4b(
        state=state,
        book_title='Tiêu Đề Thử',
        book_author='Tác Giả',
        md_text='# Chương 1\nNội dung câu 1.',
        engine_choice='VieNeu-TTS v3 Turbo (Mặc định — chất lượng cao)',
        voice_choice='Hải Đăng',
        rate_pct=0,
        clone_ref_audio=None,
        clone_ref_text='',
        force_rerender=True,
        abs_auto_push=True,
        abs_url='http://abs.lan:13378',
        abs_token='tok123',
    )

    assert mock_sync.called
    assert 'Audiobookshelf' in status
    assert 'Đã chuyển' in status


# ---------------------------------------------------------------------------
# Issue 1: cột "Ngữ cảnh" chỉ chứa đúng MỘT câu, không xuống dòng
# ---------------------------------------------------------------------------
def test_uncertainty_rows_context_is_single_line():
    state = {'chapters': [{'enabled': True, 'markdown':
        'Câu mở đầu.\nAnh ấy là MC của chương trình và CEO công ty.\n\nĐoạn khác có VIP.'}]}
    rows = app.extract_uncertainty_rows(state)
    assert rows
    for token, _cat, context, _reading in rows:
        assert '\n' not in context, f'ngữ cảnh của {token} bị xuống dòng: {context!r}'
        assert '\r' not in context
        assert len(context) <= 120
        assert token in context


def test_uncertainty_rows_context_does_not_dump_paragraph():
    state = {'chapters': [{'enabled': True, 'markdown':
        'Câu một. Câu hai nói MC. Câu ba. Câu bốn. Câu năm.'}]}
    rows = app.extract_uncertainty_rows(state)
    mc = next(r for r in rows if r[0] == 'MC')
    assert mc[2].count('.') == 1, mc[2]
    assert 'Câu một' not in mc[2], 'không được gộp câu trước'


def test_one_line_snippet_clips_long_text():
    long = 'x' * 200 + ' MC ' + 'y' * 200
    out = app._one_line_snippet(long, 'MC', limit=60)
    assert len(out) <= 60
    assert 'MC' in out
    assert '\n' not in out
    assert out.startswith('…') and out.endswith('…')


# ---------------------------------------------------------------------------
# Issue 2: cột "Đề xuất phát âm" có cách đọc thật cho viết tắt
# ---------------------------------------------------------------------------
def test_uncertainty_rows_abbr_has_real_pronunciation():
    state = {'chapters': [{'enabled': True, 'markdown': 'Anh ấy là MC và CEO.'}]}
    rows = {r[0]: r for r in app.extract_uncertainty_rows(state)}
    assert rows['MC'][3] == 'em-xi'
    assert rows['CEO'][3] == 'xi-i-ô'
    assert rows['MC'][3] != 'MC'


# ---------------------------------------------------------------------------
# Issue 3: dropdown label → tên giọng chuẩn trước khi vào TTS engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('label, expected', [
    ('Giọng kể chuyện · Thiện Minh — Nam · Bắc · Phong cách kể chuyện', 'Thiện Minh'),
    ('Giọng tuyển chọn ⭐ · Adam bựa — Nam · Bắc · Phong cách tự nhiên', 'Adam bựa'),
    ('Giọng tin tức & tự nhiên · Phạm Tuyên — Nam · Bắc', 'Phạm Tuyên'),
    ('Thiện Minh', 'Thiện Minh'),
    ('Adam bựa', 'Adam bựa'),
    ('', ''),
    (None, ''),
])
def test_extract_voice_name(label, expected):
    assert app.extract_voice_name(label) == expected


def test_extract_voice_name_handles_dash_separator():
    assert app.extract_voice_name('Giọng kể chuyện · Thái Sơn - Nam - Bắc') == 'Thái Sơn'


def test_create_tts_engine_extracts_voice_from_dropdown_label(monkeypatch):
    """Regression: trước đây cả nhãn dropdown bị truyền vào VieNeuEngine(voice=...)."""
    captured = {}

    class FakeEngine:
        def __init__(self, mode, voice=None, **kwargs):
            captured['mode'] = mode
            captured['voice'] = voice

    monkeypatch.setattr(app, 'VieNeuEngine', FakeEngine)
    label = 'Giọng kể chuyện · Thiện Minh — Nam · Bắc · Phong cách kể chuyện'
    app._create_tts_engine(
        'VieNeu-TTS v3 Turbo (Mặc định — chất lượng cao)', label, 0
    )
    assert captured['voice'] == 'Thiện Minh'
    assert '·' not in captured['voice'] and '—' not in captured['voice']


def test_create_tts_engine_keeps_clone_and_classic_routing(monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, mode, voice=None, **kwargs):
            captured['mode'] = mode
            captured['voice'] = voice

    monkeypatch.setattr(app, 'VieNeuEngine', FakeEngine)
    monkeypatch.setattr(app, '_clone_reference_for_label', lambda _l: ('ref.wav', 'text'))

    app._create_tts_engine('VieNeu-TTS v3 Turbo', 'Clone · Vui Vẻ', 0)
    assert captured['mode'] == 'clone'

    app._create_tts_engine('VieNeu-TTS v3 Turbo', 'Nguyễn Ngọc Ngạn (kinh điển)', 0)
    assert captured['mode'] == 'classic'
