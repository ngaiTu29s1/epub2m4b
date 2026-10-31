import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any
import requests

logger = logging.getLogger('epub2m4b.abs')


def load_env_file(dotenv_path: Path | str | None = None) -> dict[str, str]:
    """
    Load environment variables from a .env file into os.environ (if not already set).
    Pure python implementation without external dependency.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / '.env'
    else:
        dotenv_path = Path(dotenv_path)

    loaded = {}
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return loaded

    try:
        content = dotenv_path.read_text(encoding='utf-8')
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                loaded[key] = val
                if key not in os.environ:
                    os.environ[key] = val
    except Exception as exc:
        logger.warning(f'Không thể đọc file .env: {exc}')

    return loaded


# Auto-load .env at module import
load_env_file()


def _sanitize_folder_name(name: str) -> str:
    """Sanitize directory or file name for filesystem compatibility."""
    clean = re.sub(r'[\\/*?:"<>|]', '', name).strip()
    return clean or 'Unknown'


class ABSClient:
    """
    Audiobookshelf API Client.
    Supports both direct network storage copy (SMB/NFS) and HTTP REST upload,
    followed by triggering library scan.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        library_id: str | None = None,
        storage_path: str | Path | None = None,
        auto_scan: bool | str | None = None,
        timeout: float = 30.0,
    ):
        raw_url = base_url if base_url is not None else os.getenv('ABS_URL', '')
        self.base_url = raw_url.strip().rstrip('/')
        self.token = (token if token is not None else os.getenv('ABS_TOKEN', '')).strip()
        self.library_id = (
            library_id if library_id is not None else os.getenv('ABS_LIBRARY_ID', '')
        ).strip()

        raw_storage = (
            storage_path
            if storage_path is not None
            else os.getenv('ABS_STORAGE_PATH', '')
        )
        self.storage_path = Path(raw_storage) if raw_storage else None

        if auto_scan is not None:
            if isinstance(auto_scan, bool):
                self.auto_scan = auto_scan
            else:
                self.auto_scan = str(auto_scan).lower() in ('1', 'true', 'yes')
        else:
            self.auto_scan = os.getenv('ABS_AUTO_SCAN', 'true').lower() in ('1', 'true', 'yes')

        self.timeout = timeout

    def is_configured(self) -> bool:
        """Check if minimum required credentials for REST API are present."""
        return bool(self.base_url and self.token)

    def _headers(self) -> dict[str, str]:
        headers = {'Accept': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers

    def check_connection(self) -> dict[str, Any]:
        """
        Test connection and authentication to Audiobookshelf server.
        Returns:
            {'connected': bool, 'username': str, 'server_version': str, 'error': str | None}
        """
        if not self.base_url:
            return {'connected': False, 'error': 'Chưa nhập địa chỉ ABS_URL', 'status_code': None}
        if not self.token:
            return {'connected': False, 'error': 'Chưa nhập token ABS_TOKEN', 'status_code': None}

        url = f'{self.base_url}/api/me'
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                username = data.get('user', {}).get('username', 'User')
                version = data.get('serverVersion', 'Unknown')
                return {
                    'connected': True,
                    'username': username,
                    'server_version': version,
                    'status_code': 200,
                    'error': None,
                }
            elif resp.status_code in (401, 403):
                return {
                    'connected': False,
                    'error': f'Xác thực thất bại (Mã lỗi {resp.status_code}). Vui lòng kiểm tra lại ABS_TOKEN.',
                    'status_code': resp.status_code,
                }
            else:
                # Fallback check /api/libraries
                lib_resp = requests.get(
                    f'{self.base_url}/api/libraries',
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                if lib_resp.status_code == 200:
                    return {
                        'connected': True,
                        'username': 'API User',
                        'server_version': 'Unknown',
                        'status_code': 200,
                        'error': None,
                    }
                return {
                    'connected': False,
                    'error': f'Lỗi phản hồi từ máy chủ: HTTP {resp.status_code}',
                    'status_code': resp.status_code,
                }
        except requests.exceptions.Timeout:
            return {'connected': False, 'error': f'Hết thời gian chờ kết nối ({self.timeout}s)', 'status_code': None}
        except requests.exceptions.ConnectionError:
            return {'connected': False, 'error': f'Không thể kết nối đến máy chủ tại {self.base_url}', 'status_code': None}
        except Exception as exc:
            return {'connected': False, 'error': f'Lỗi không xác định: {exc}', 'status_code': None}

    def get_libraries(self) -> list[dict[str, Any]]:
        """Fetch list of libraries configured on Audiobookshelf."""
        if not self.is_configured():
            logger.warning('ABS chưa được cấu hình URL hoặc Token.')
            return []

        url = f'{self.base_url}/api/libraries'
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get('libraries', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        except Exception as exc:
            logger.error(f'Không thể lấy danh sách thư viện ABS: {exc}')
            return []

    def find_audiobook_library_id(self) -> str | None:
        """ID của thư viện sách nói/ebook đầu tiên, hoặc ``None`` nếu không có.

        KHÔNG fallback về thư viện đầu tiên: server hay đặt Podcast trước, đẩy
        nhầm sách vào đó làm hỏng thư viện. Thà trả ``None`` và báo lỗi rõ ràng.
        """
        libs = self.get_libraries()
        if not libs:
            return None

        for lib in libs:
            media_type = str(lib.get('mediaType', '')).lower()
            if media_type in ('book', 'books', 'audiobook', 'audiobooks'):
                return lib.get('id')
        return None

    def copy_to_storage(
        self,
        file_path: Path,
        author: str,
        title: str,
        storage_path: Path | None = None,
    ) -> Path:
        """
        Copy audiobook file directly to shared storage directory (SMB/NFS)
        following Audiobookshelf format: {storage_path}/{author}/{title}/{title}.m4b

        Tên file đích luôn là ``{title}.m4b`` (không dùng tên file nguồn) để ABS
        nhận đúng sách: thư mục ``Author/Title`` + file cùng tên ``Title``.
        """
        base_dir = Path(storage_path or self.storage_path or '')
        if not str(base_dir):
            raise ValueError('Chưa cấu hình storage_path để copy file trực tiếp.')

        clean_author = _sanitize_folder_name(author or 'Tác Giả Khác')
        clean_title = _sanitize_folder_name(title or file_path.stem)

        dest_dir = base_dir / clean_author / clean_title
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_file = dest_dir / f'{clean_title}{file_path.suffix or ".m4b"}'
        logger.info(f'📁 [ABS STORAGE] Đang sao chép file sang: {dest_file}')
        shutil.copy2(file_path, dest_file)
        logger.info(f'✅ [ABS STORAGE] Sao chép hoàn tất: {dest_file.name}')
        return dest_file

    def upload_file(self, file_path: Path, library_id: str | None = None) -> dict[str, Any]:
        """
        Upload audiobook file via REST API multipart upload.
        Endpoint: POST /api/libraries/{id}/upload
        """
        if not self.is_configured():
            return {'success': False, 'error': 'ABS chưa được cấu hình URL hoặc Token'}

        file_path = Path(file_path)
        if not file_path.exists():
            return {'success': False, 'error': f'File không tồn tại: {file_path}'}

        target_lib = library_id or self.library_id or self.find_audiobook_library_id()
        if not target_lib:
            return {'success': False, 'error': 'Không tìm thấy ID thư viện Audiobookshelf để upload'}

        url = f'{self.base_url}/api/libraries/{target_lib}/upload'
        logger.info(f'📤 [ABS UPLOAD] Đang tải lên {file_path.name} tới thư viện {target_lib}...')

        try:
            with open(file_path, 'rb') as f:
                files = {'files': (file_path.name, f, 'audio/x-m4b')}
                resp = requests.post(url, headers={'Authorization': f'Bearer {self.token}'}, files=files, timeout=300.0)
                if resp.status_code in (200, 201, 204):
                    logger.info('✅ [ABS UPLOAD] Tải lên thành công!')
                    return {
                        'success': True,
                        'library_id': target_lib,
                        'status_code': resp.status_code,
                        'data': resp.json() if resp.text and resp.headers.get('content-type', '').startswith('application/json') else {},
                    }
                else:
                    err_msg = f'Lỗi upload HTTP {resp.status_code}: {resp.text[:200]}'
                    logger.error(f'❌ [ABS UPLOAD] {err_msg}')
                    return {'success': False, 'error': err_msg, 'status_code': resp.status_code}
        except Exception as exc:
            logger.exception(f'❌ [ABS UPLOAD] Lỗi khi upload file: {exc}')
            return {'success': False, 'error': str(exc)}

    def trigger_scan(self, library_id: str | None = None) -> dict[str, Any]:
        """
        Trigger library scan on Audiobookshelf.
        Endpoint: POST /api/libraries/{id}/scan
        """
        if not self.is_configured():
            return {'success': False, 'error': 'ABS chưa được cấu hình'}

        target_lib = library_id or self.library_id or self.find_audiobook_library_id()
        if not target_lib:
            return {'success': False, 'error': 'Không tìm thấy ID thư viện để scan'}

        url = f'{self.base_url}/api/libraries/{target_lib}/scan'
        try:
            resp = requests.post(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code in (200, 201, 204):
                logger.info(f'🔄 [ABS SCAN] Đã kích hoạt quét thư viện {target_lib} thành công.')
                return {'success': True, 'library_id': target_lib, 'status_code': resp.status_code}
            return {
                'success': False,
                'error': f'Lỗi quét thư viện HTTP {resp.status_code}: {resp.text[:200]}',
                'status_code': resp.status_code,
            }
        except Exception as exc:
            logger.error(f'❌ [ABS SCAN] Lỗi khi gọi lệnh scan: {exc}')
            return {'success': False, 'error': str(exc)}

    def sync_audiobook(
        self,
        file_path: Path,
        author: str,
        title: str,
        library_id: str | None = None,
        auto_scan: bool | None = None,
    ) -> dict[str, Any]:
        """
        High-level sync orchestrator:
        1. Checks if storage_path is mounted/accessible -> Direct copy.
        2. Otherwise if API configured -> REST HTTP upload.
        3. If successful and auto_scan enabled -> Trigger scan.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {'success': False, 'error': f'File không tồn tại: {file_path}'}

        should_scan = auto_scan if auto_scan is not None else self.auto_scan
        target_lib = library_id or self.library_id

        # 1. Direct storage copy attempt if storage_path is specified and exists
        if self.storage_path and self.storage_path.exists():
            try:
                dest = self.copy_to_storage(file_path, author=author, title=title)
                scan_res = self.trigger_scan(target_lib) if should_scan and self.is_configured() else {'success': True}
                return {
                    'success': True,
                    'method': 'direct_storage_copy',
                    'destination': str(dest),
                    'scanned': scan_res.get('success', False),
                    'scan_detail': scan_res,
                }
            except Exception as exc:
                logger.warning(f'Không thể copy trực tiếp vào storage ({exc}), thử chuyển sang REST upload...')

        # 2. REST API upload attempt
        if self.is_configured():
            upload_res = self.upload_file(file_path, target_lib)
            if not upload_res.get('success'):
                return upload_res
            scan_res = self.trigger_scan(target_lib) if should_scan else {'success': True}
            return {
                'success': True,
                'method': 'rest_upload',
                'upload_detail': upload_res,
                'scanned': scan_res.get('success', False),
                'scan_detail': scan_res,
            }

        return {
            'success': False,
            'error': 'Không có cấu hình Audiobookshelf hợp lệ (cần URL + Token hoặc Thư mục Mount).',
        }
