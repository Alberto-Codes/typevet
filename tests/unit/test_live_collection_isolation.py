"""Live tests must not probe the router during default pytest collection or runs."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

LIVE_MODULE = "tests.live.test_gemma4_llama_cpp"


@pytest.mark.unit
def test_live_module_import_never_probes_router() -> None:
    probe_calls: list[object] = []

    def _fail_probe(*args: object, **kwargs: object) -> object:
        probe_calls.append((args, kwargs))
        raise AssertionError("live router probe during import/collection")

    with patch("httpx.get", _fail_probe):
        sys.modules.pop(LIVE_MODULE, None)
        importlib.import_module(LIVE_MODULE)
    assert probe_calls == []


@pytest.mark.unit
def test_default_pytest_collect_never_probes_live_router() -> None:
    probe_calls: list[object] = []

    def _fail_probe(*args: object, **kwargs: object) -> object:
        probe_calls.append((args, kwargs))
        raise AssertionError("live router probe during default pytest collection")

    with patch("httpx.get", _fail_probe):
        exit_code = pytest.main(["-q", "-m", "not live", "--collect-only"])
    assert probe_calls == []
    assert exit_code == pytest.ExitCode.OK


@pytest.mark.unit
def test_default_pytest_run_never_probes_live_router() -> None:
    probe_calls: list[object] = []

    def _fail_probe(*args: object, **kwargs: object) -> object:
        probe_calls.append((args, kwargs))
        raise AssertionError("live router probe during default pytest run")

    with patch("httpx.get", _fail_probe):
        exit_code = pytest.main(
            [
                "-q",
                "-m",
                "not live",
                "--ignore=tests/unit/test_live_collection_isolation.py",
            ]
        )
    assert probe_calls == []
    assert exit_code == pytest.ExitCode.OK
