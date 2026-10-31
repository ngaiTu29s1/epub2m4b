from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests
from core.abs_client import ABSClient, _sanitize_folder_name, load_env_file


def test_sanitize_folder_name():
    assert _sanitize_folder_name('Tác Giả: Nguyễn Du / 1800?') == 'Tác Giả Nguyễn Du  1800'
    assert _sanitize_folder_name('   ') == 'Unknown'
    assert _sanitize_folder_name('Đứa Trẻ Hư (Bản Chuẩn)') == 'Đứa Trẻ Hư (Bản Chuẩn)'


def test_load_env_file(tmp_path: Path, monkeypatch):
    env_file = tmp_path / '.env'
    env_file.write_text(
        '# Cấu hình thử nghiệm\n'
        'TEST_KEY_1=value1\n'
        'TEST_KEY_2="quoted_value"\n'
        "TEST_KEY_3='single_quoted'\n"
        'INVALID LINE\n',
        encoding='utf-8',
    )
    monkeypatch.delenv('TEST_KEY_1', raising=False)
    monkeypatch.delenv('TEST_KEY_2', raising=False)
    monkeypatch.delenv('TEST_KEY_3', raising=False)

    loaded = load_env_file(env_file)
    assert loaded['TEST_KEY_1'] == 'value1'
    assert loaded['TEST_KEY_2'] == 'quoted_value'
    assert loaded['TEST_KEY_3'] == 'single_quoted'


def test_abs_client_init_and_config():
    client = ABSClient(
        base_url='http://192.168.1.50:13378/',
        token='secret-token',
        library_id='lib-123',
        storage_path='/mnt/audiobooks',
        auto_scan=True,
    )
    assert client.base_url == 'http://192.168.1.50:13378'
    assert client.token == 'secret-token'
    assert client.library_id == 'lib-123'
    assert client.is_configured() is True
    assert client._headers()['Authorization'] == 'Bearer secret-token'

    unconfigured = ABSClient(base_url='', token='')
    assert unconfigured.is_configured() is False


@patch('requests.get')
def test_check_connection_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        'user': {'username': 'tu_homelab'},
        'serverVersion': '2.8.0',
    }
    mock_get.return_value = mock_resp

    client = ABSClient(base_url='http://abs.local', token='valid-token')
    res = client.check_connection()

    assert res['connected'] is True
    assert res['username'] == 'tu_homelab'
    assert res['server_version'] == '2.8.0'
    assert res['error'] is None


@patch('requests.get')
def test_check_connection_auth_failure(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_get.return_value = mock_resp

    client = ABSClient(base_url='http://abs.local', token='bad-token')
    res = client.check_connection()

    assert res['connected'] is False
    assert res['status_code'] == 401
    assert 'Xác thực thất bại' in res['error']


@patch('requests.get')
def test_check_connection_offline(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError('Connection refused')

    client = ABSClient(base_url='http://abs.local', token='valid-token')
    res = client.check_connection()

    assert res['connected'] is False
    assert 'Không thể kết nối' in res['error']


def test_check_connection_missing_credentials():
    client = ABSClient(base_url='', token='')
    res = client.check_connection()
    assert res['connected'] is False
    assert 'Chưa nhập' in res['error']


@patch('requests.get')
def test_get_libraries_and_find_audiobook_library(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        'libraries': [
            {'id': 'lib_podcasts', 'name': 'Podcasts', 'mediaType': 'podcast'},
            {'id': 'lib_audiobooks', 'name': 'Sách Nói Hay', 'mediaType': 'book'},
        ]
    }
    mock_get.return_value = mock_resp

    client = ABSClient(base_url='http://abs.local', token='valid-token')
    libs = client.get_libraries()
    assert len(libs) == 2

    found_id = client.find_audiobook_library_id()
    assert found_id == 'lib_audiobooks'


def test_find_audiobook_library_returns_none_without_book_library(monkeypatch):
    """Không có thư viện sách nào → None, KHÔNG fallback về Podcast."""
    client = ABSClient(base_url='http://abs.local', token='t')
    monkeypatch.setattr(client, 'get_libraries', lambda: [
        {'id': 'lib_podcasts', 'name': 'Podcasts', 'mediaType': 'podcast'},
        {'id': 'lib_music', 'name': 'Nhạc', 'mediaType': 'music'},
    ])
    assert client.find_audiobook_library_id() is None


def test_find_audiobook_library_accepts_audiobook_media_type(monkeypatch):
    client = ABSClient(base_url='http://abs.local', token='t')
    monkeypatch.setattr(client, 'get_libraries', lambda: [
        {'id': 'lib_pod', 'mediaType': 'podcast'},
        {'id': 'lib_ab', 'mediaType': 'audiobook'},
    ])
    assert client.find_audiobook_library_id() == 'lib_ab'


def test_find_audiobook_library_no_libraries(monkeypatch):
    client = ABSClient(base_url='http://abs.local', token='t')
    monkeypatch.setattr(client, 'get_libraries', lambda: [])
    assert client.find_audiobook_library_id() is None


def test_copy_to_storage(tmp_path: Path):
    storage_root = tmp_path / 'nas_audiobooks'
    source_file = tmp_path / 'cuon_sach.m4b'
    source_file.write_bytes(b'M4B_MOCK_DATA')

    client = ABSClient(storage_path=storage_root)
    dest = client.copy_to_storage(
        file_path=source_file,
        author='Tử Kim Trần',
        title='Đứa Trẻ Hư',
    )

    # Chuẩn ABS: {storage}/{author}/{title}/{title}.m4b — KHÔNG dùng tên file nguồn.
    expected_path = storage_root / 'Tử Kim Trần' / 'Đứa Trẻ Hư' / 'Đứa Trẻ Hư.m4b'
    assert dest == expected_path
    assert dest.exists()
    assert dest.read_bytes() == b'M4B_MOCK_DATA'
    assert not (storage_root / 'Tử Kim Trần' / 'Đứa Trẻ Hư' / 'cuon_sach.m4b').exists()


@patch('requests.post')
def test_upload_file_success(mock_post, tmp_path: Path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '{"success": true}'
    mock_resp.headers = {'content-type': 'application/json'}
    mock_resp.json.return_value = {'success': True}
    mock_post.return_value = mock_resp

    test_file = tmp_path / 'test.m4b'
    test_file.write_bytes(b'AUDIO')

    client = ABSClient(base_url='http://abs.local', token='token123', library_id='lib_1')
    res = client.upload_file(test_file)

    assert res['success'] is True
    assert res['library_id'] == 'lib_1'
    assert mock_post.called


@patch('requests.post')
def test_trigger_scan_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_post.return_value = mock_resp

    client = ABSClient(base_url='http://abs.local', token='token123', library_id='lib_1')
    res = client.trigger_scan()

    assert res['success'] is True
    assert res['library_id'] == 'lib_1'


def test_sync_audiobook_via_storage_copy(tmp_path: Path):
    storage_dir = tmp_path / 'abs_storage'
    storage_dir.mkdir()
    source_file = tmp_path / 'sample.m4b'
    source_file.write_bytes(b'AUDIO_CONTENT')

    client = ABSClient(storage_path=storage_dir, auto_scan=False)
    res = client.sync_audiobook(
        file_path=source_file,
        author='Nguyễn Nhật Ánh',
        title='Tôi Thấy Hoa Vàng',
    )

    assert res['success'] is True
    assert res['method'] == 'direct_storage_copy'
    assert Path(res['destination']).exists()


@patch('core.abs_client.ABSClient.upload_file')
@patch('core.abs_client.ABSClient.trigger_scan')
def test_sync_audiobook_via_upload(mock_scan, mock_upload, tmp_path: Path):
    source_file = tmp_path / 'sample.m4b'
    source_file.write_bytes(b'AUDIO_CONTENT')

    mock_upload.return_value = {'success': True}
    mock_scan.return_value = {'success': True}

    client = ABSClient(base_url='http://abs.local', token='tok', library_id='lib_book', auto_scan=True)
    res = client.sync_audiobook(
        file_path=source_file,
        author='Nam Cao',
        title='Chí Phèo',
    )

    assert res['success'] is True
    assert res['method'] == 'rest_upload'
    assert res['scanned'] is True


# ---------------------------------------------------------------------------
# Hardening: không được rò request thật ra LAN khi chưa cấu hình library_id
# ---------------------------------------------------------------------------
def test_sync_without_library_id_does_not_touch_network(tmp_path: Path):
    """Khi thiếu library_id và không cấu hình gì, phải fail sạch (không gọi mạng)."""
    source = tmp_path / 'a.m4b'
    source.write_bytes(b'AUDIO')

    client = ABSClient(base_url='', token='', storage_path=None, auto_scan=False)
    with patch('requests.get', side_effect=AssertionError('không được gọi mạng')) as g, \
         patch('requests.post', side_effect=AssertionError('không được gọi mạng')):
        res = client.sync_audiobook(source, author='A', title='T')
    assert res['success'] is False
    assert 'error' in res
    assert not g.called


def test_upload_file_autodetects_library_when_missing(tmp_path: Path):
    """Không có library_id → tự dò thư viện qua get_libraries (mock)."""
    source = tmp_path / 'b.m4b'
    source.write_bytes(b'AUDIO')

    # library_id='' để không kế thừa ABS_LIBRARY_ID từ .env thật của máy dev.
    client = ABSClient(base_url='http://abs.local', token='t', library_id='')
    ok = MagicMock(status_code=200, text='{}')
    ok.headers = {'content-type': 'application/json'}
    ok.json.return_value = {}

    with patch.object(client, 'find_audiobook_library_id', return_value='lib_auto') as finder, \
         patch('requests.post', return_value=ok) as post:
        res = client.upload_file(source)

    assert finder.called
    assert res['success'] is True
    assert res['library_id'] == 'lib_auto'
    assert '/api/libraries/lib_auto/upload' in post.call_args[0][0]


def test_upload_file_missing_file_returns_error(tmp_path: Path):
    client = ABSClient(base_url='http://abs.local', token='t', library_id='L')
    res = client.upload_file(tmp_path / 'nope.m4b')
    assert res['success'] is False
    assert 'không tồn tại' in res['error'].lower()


def test_trigger_scan_http_error_is_graceful():
    client = ABSClient(base_url='http://abs.local', token='t', library_id='L')
    with patch('requests.post', return_value=MagicMock(status_code=500, text='boom')):
        res = client.trigger_scan()
    assert res['success'] is False
    assert res['status_code'] == 500


def test_sync_falls_back_to_upload_when_storage_copy_fails(tmp_path: Path):
    """storage_path có nhưng copy lỗi → chuyển sang REST upload thay vì crash."""
    source = tmp_path / 'c.m4b'
    source.write_bytes(b'AUDIO')
    storage = tmp_path / 'store'
    storage.mkdir()

    client = ABSClient(
        base_url='http://abs.local', token='t', library_id='L',
        storage_path=storage, auto_scan=False,
    )
    with patch.object(client, 'copy_to_storage', side_effect=OSError('disk full')), \
         patch.object(client, 'upload_file', return_value={'success': True}) as up:
        res = client.sync_audiobook(source, author='A', title='T')

    assert up.called
    assert res['success'] is True
    assert res['method'] == 'rest_upload'


def test_sync_unconfigured_storage_missing_returns_error(tmp_path: Path):
    """storage_path không tồn tại + không có API → lỗi rõ ràng, không exception."""
    source = tmp_path / 'd.m4b'
    source.write_bytes(b'AUDIO')
    client = ABSClient(base_url='', token='', storage_path=tmp_path / 'missing', auto_scan=False)
    res = client.sync_audiobook(source, author='A', title='T')
    assert res['success'] is False
    assert res.get('error')


def test_sync_missing_file_returns_error(tmp_path: Path):
    client = ABSClient(base_url='http://abs.local', token='t', library_id='L')
    res = client.sync_audiobook(tmp_path / 'ghost.m4b', author='A', title='T')
    assert res['success'] is False
    assert 'không tồn tại' in res['error'].lower()
