"""Unit tests that keep secrets out of pytest failure output (#251).

A live test once failed and pytest printed the pod API key, because the
repository ``addopts`` enabled ``--showlocals``. These tests run a small
failing test under the repository's own ``addopts`` and check the output.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SENTINEL_ENV = "TYPEVET_TEST_SENTINEL_KEY"
_SENTINEL = "sk-sentinel-3f9c1a7e5b2d4086"

_FAILING_TEST = f"""
import os


def test_fails_with_key_in_local():
    api_key = os.environ["{_SENTINEL_ENV}"]
    raise AssertionError("request refused")
"""


def _repo_addopts() -> list[str]:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    return list(config["tool"]["pytest"]["ini_options"]["addopts"])


def test_repo_addopts_do_not_print_a_key_held_in_a_local(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SENTINEL_ENV, _SENTINEL)
    pytester.makepyfile(test_leak=_FAILING_TEST)

    result = pytester.runpytest_subprocess(*_repo_addopts())

    result.assert_outcomes(failed=1)
    output = result.stdout.str() + result.stderr.str()
    assert "request refused" in output
    assert _SENTINEL not in output
