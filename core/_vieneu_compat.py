"""Compatibility shims for the ``vieneu`` / ``neucodec`` stack.

Two independent packaging conflicts exist in this venv (torch 2.14, torchao
0.18, torchtune 0.6.1):

1. ``torchtune 0.6.1`` imports ``torchao.dtypes.nf4tensor``, which torchao 0.18
   no longer ships; importing ``torchtune.modules`` therefore raises
   ``ModuleNotFoundError``. ``neucodec`` only needs
   ``RotaryPositionalEmbeddings`` from that package.
2. Fixing (1) by pinning torchao would either break against torch 2.14 (0.12)
   or force a torch upgrade (torchao 0.18 requires a newer torchtune). Neither
   is acceptable: upgrading torch is forbidden by the migration constraints.

The shim below loads ``torchtune/modules/position_embeddings.py`` directly from
disk (it only imports ``torch``/``nn``) and registers a synthetic
``torchtune.modules`` in ``sys.modules`` so ``neucodec`` can import from it
without executing the broken ``torchtune/__init__.py`` chain.

Call :func:`ensure_neucodec_importable` before importing ``neucodec``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_SHIM_FLAG = '__epub2m4b_torchtune_shim__'


def _site_packages() -> Path | None:
    for p in sys.path:
        if p.endswith('site-packages'):
            return Path(p)
    return None


def _load_rotary_module(site: Path):
    """Load ``torchtune/modules/position_embeddings.py`` standalone."""
    pos_file = site / 'torchtune' / 'modules' / 'position_embeddings.py'
    if not pos_file.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        'torchtune.modules.position_embeddings', pos_file
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_torchtune_modules_works() -> bool:
    try:
        from torchtune.modules import RotaryPositionalEmbeddings  # noqa: F401
    except Exception:
        return False
    return True


def ensure_neucodec_importable() -> bool:
    """Make ``from neucodec import ...`` work in this venv.

    Returns ``True`` when ``neucodec`` is importable afterwards (either natively
    or through the shim), ``False`` when it is not installed at all.
    """
    if 'neucodec' in sys.modules:
        return True

    if not _real_torchtune_modules_works():
        site = _site_packages()
        if site is None:
            return False
        rotary_mod = _load_rotary_module(site)
        if rotary_mod is None:
            return False

        pkg = types.ModuleType('torchtune')
        pkg.__path__ = [str(site / 'torchtune')]
        modules = types.ModuleType('torchtune.modules')
        modules.__path__ = [str(site / 'torchtune' / 'modules')]
        modules.RotaryPositionalEmbeddings = rotary_mod.RotaryPositionalEmbeddings
        modules.__epub2m4b_shim__ = True
        pkg.modules = modules
        sys.modules['torchtune'] = pkg
        sys.modules['torchtune.modules'] = modules
        sys.modules['torchtune.modules.position_embeddings'] = rotary_mod

    sys.modules[_SHIM_FLAG] = types.ModuleType(_SHIM_FLAG)

    try:
        import neucodec  # noqa: F401
    except ModuleNotFoundError:
        return False
    except Exception:
        return False
    return True
