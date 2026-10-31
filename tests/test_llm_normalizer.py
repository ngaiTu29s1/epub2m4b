"""Tests for the Self-Enriching Hybrid Normalizer (cache + LLM fallback)."""
import json

import pytest

from core import llm_normalizer as ln
from core.text_normalizer import normalize_vi_text


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Mỗi test dùng runtime cache riêng trong tmp_path, không đụng workspace."""
    monkeypatch.setattr(ln, '_RUNTIME_CACHE_PATH', tmp_path / 'dict.json')
    ln.reload_cache()
    yield
    ln.reload_cache()


def test_seed_cache_contains_chemical_and_math():
    assert ln.lookup('H2SO4') == 'H hai S O bốn'
    assert ln.lookup('h2so4') == 'H hai S O bốn'
    assert ln.lookup('CO2') == 'C O hai'
    assert ln.lookup('NaCl') == 'N A C L'
    assert ln.lookup('E = mc^2') == 'E bằng m c bình phương'
    assert 'bình phương' in ln.lookup('x^2 + y^2 = z^2')


def test_lookup_returns_none_for_unknown():
    assert ln.lookup('không-có-trong-cache-xyz') is None


def test_cache_lookup_is_case_insensitive():
    assert ln.lookup('co2') == ln.lookup('CO2') == 'C O hai'


def test_add_entry_persists_and_reloads(tmp_path):
    target = tmp_path / 'dict.json'
    ln.add_entry('Zx9', 'Z x chín', kind='chemical')
    assert target.exists()
    payload = json.loads(target.read_text(encoding='utf-8'))
    # Runtime cache ghi khoá chuẩn hoá; giữ ``literal`` gốc bên trong.
    entry = payload['entries']['zx9']
    assert entry['spoken'] == 'Z x chín'
    assert entry['literal'] == 'Zx9'
    ln.reload_cache()
    assert ln.lookup('Zx9') == 'Z x chín'


def test_enrich_normalizes_chemical_formula():
    out = ln.enrich_and_normalize('Nước là H2O và khí CO2.')
    assert 'H hai O' in out
    assert 'C O hai' in out


def test_enrich_is_graceful_without_api_key(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    assert ln.llm_available() is False
    # Không được crash kể cả với biểu thức lạ.
    out = ln.enrich_and_normalize('Phương trình x^2 + 2x + 1 = 0 rất thú vị.')
    assert isinstance(out, str)
    assert out  # vẫn trả nội dung rule-based


def test_enrich_never_crashes_on_garbage(monkeypatch):
    monkeypatch.setattr(ln, '_enrich_impl', lambda _t: (_ for _ in ()).throw(RuntimeError('boom')))
    assert ln.enrich_and_normalize('bất kỳ') == 'bất kỳ'


def test_llm_call_returns_none_without_key(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    assert ln._call_llm('test') is None


def test_add_entry_noop_on_empty():
    ln.add_entry('', '')
    assert ln.lookup('') is None


def test_normalize_vi_text_uses_cache_for_math():
    out = normalize_vi_text('Theo E = mc^2 thì năng lượng rất lớn.')
    assert 'bình phương' in out
