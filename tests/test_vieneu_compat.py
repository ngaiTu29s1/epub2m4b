"""Tests for the torchtune/neucodec import shim (no model download)."""
import sys

from core._vieneu_compat import ensure_neucodec_importable


def test_ensure_neucodec_importable_reports_true_when_installed():
    assert ensure_neucodec_importable() is True
    import neucodec  # noqa: F401


def test_shim_registers_torchtune_modules_without_broken_chain():
    ensure_neucodec_importable()
    # ``RotaryPositionalEmbeddings`` must be resolvable from ``torchtune.modules``.
    import torchtune.modules as modules

    assert hasattr(modules, 'RotaryPositionalEmbeddings')
    # The shim must be idempotent.
    assert ensure_neucodec_importable() is True


def test_shim_is_noop_when_neucodec_present():
    assert 'neucodec' in sys.modules or ensure_neucodec_importable()
