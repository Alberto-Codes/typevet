"""Unit tests for the live backend guard in the image live tests (#407)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from typevet_evals.runner.live_gate import live_backend


@pytest.mark.unit
def test_live_backend_rejects_fake_before_any_http() -> None:
    """``fake`` fails with the accepted live values and makes no HTTP call."""
    with (
        patch("typevet_evals.runner.live_gate.httpx") as client,
        pytest.raises(ValueError, match="TYPEVET_BACKEND") as caught,
    ):
        live_backend({"TYPEVET_BACKEND": "fake"})

    message = str(caught.value)
    assert "llama_cpp" in message
    assert "vllm" in message
    assert "fake" in message
    assert client.mock_calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", "llama_cpp"), ("llama_cpp", "llama_cpp"), ("vllm", "vllm")],
    ids=["unset", "llama-cpp", "vllm"],
)
def test_live_backend_returns_live_values(raw: str, expected: str) -> None:
    """The live values pass through unchanged."""
    assert live_backend({"TYPEVET_BACKEND": raw}) == expected


@pytest.mark.unit
def test_live_backend_keeps_load_backend_error_for_unknown_value() -> None:
    """An unknown value still fails with the ``load_backend`` message."""
    with pytest.raises(ValueError, match="TYPEVET_BACKEND must be"):
        live_backend({"TYPEVET_BACKEND": "other"})
