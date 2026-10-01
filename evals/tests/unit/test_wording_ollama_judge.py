"""Unit checks for the Ollama judge of the live wording evolution (#333).

The live test reads ``TYPEVET_WORDING_JUDGE_PROVIDER=ollama``,
``TYPEVET_OLLAMA_BASE`` and ``TYPEVET_WORDING_JUDGE``. These checks drive its
helpers with a scripted ``environ``, scripted Ollama JSON and a recording
adapter class, so no call leaves the process.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_ollama_judge.py
    ```

See Also:
    - [typevet_evals.wording.calls][]: the call records of the judge port
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, ClassVar, Self

import pytest

from evals.tests.live import test_wording_evolution_live as live

pytestmark = pytest.mark.unit

BASE = "http://localhost:11434"
DIGEST = "24e550a1" + "0" * 51 + "67e0c"
VERSION = {"version": "0.35.0"}
TAGS = {
    "models": [
        {"name": "other:7b", "digest": "f" * 64, "size": 1, "details": {}},
        {"name": "nimble:latest", "digest": DIGEST, "size": 2, "details": {}},
    ]
}


def _getter(
    responses: Mapping[str, Mapping[str, Any]], seen: list[str] | None = None
) -> Any:
    def get(url: str) -> Mapping[str, Any]:
        if seen is not None:
            seen.append(url)
        return responses[url]

    return get


def _ollama_get(base: str = BASE, seen: list[str] | None = None) -> Any:
    return _getter({f"{base}/api/version": VERSION, f"{base}/api/tags": TAGS}, seen)


class _RecordingAdapter:
    """Stand in for ``HTTPSystemOneAdapter``; record the constructor keywords."""

    built: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).built.append(kwargs)
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_adapter() -> None:
    _RecordingAdapter.built = []


def test_provider_knob_accepts_ollama() -> None:
    environ = {"TYPEVET_WORDING_JUDGE_PROVIDER": "ollama"}
    assert live._judge_provider(environ) == "ollama"


@pytest.mark.parametrize("value", ["", "gemma", "jev"])
def test_provider_knob_keeps_gemma_default_and_jev(value: str) -> None:
    provider = live._judge_provider({"TYPEVET_WORDING_JUDGE_PROVIDER": value})
    assert provider == (value or "gemma")


def test_provider_knob_refuses_unknown_provider() -> None:
    environ = {"TYPEVET_WORDING_JUDGE_PROVIDER": "openrouter"}
    with pytest.raises(pytest.fail.Exception, match="must be one of"):
        live._judge_provider(environ)


def test_ollama_concurrency_default_is_one() -> None:
    assert live._DEFAULT_CONCURRENCY["ollama"] == 1


@pytest.mark.parametrize(
    ("provider", "expected"), [("ollama", "nimble"), ("gemma", live._JUDGE)]
)
def test_judge_name_defaults_per_provider(provider: str, expected: str) -> None:
    assert live.judge_name({}, provider) == expected
    assert live.judge_name({"TYPEVET_WORDING_JUDGE": "  "}, provider) == expected


@pytest.mark.parametrize("provider", ["ollama", "gemma"])
def test_judge_name_reads_the_judge_knob(provider: str) -> None:
    environ = {"TYPEVET_WORDING_JUDGE": " qwen3:8b "}
    assert live.judge_name(environ, provider) == "qwen3:8b"


def test_ollama_identity_reads_version_and_digest() -> None:
    seen: list[str] = []
    identity = live.ollama_identity(_ollama_get(seen=seen), BASE, "nimble")
    assert identity == {"ollama_version": "0.35.0", "model_digest": DIGEST}
    assert seen == [f"{BASE}/api/version", f"{BASE}/api/tags"]


def test_ollama_identity_matches_an_exact_tag() -> None:
    identity = live.ollama_identity(_ollama_get(), BASE + "/", "other:7b")
    assert identity["model_digest"] == "f" * 64


def test_ollama_identity_refuses_an_unlisted_model_value_free() -> None:
    with pytest.raises(ValueError, match="not listed") as caught:
        live.ollama_identity(_ollama_get(), BASE, "absent")
    message = str(caught.value)
    assert "other" not in message
    assert "nimble" not in message
    assert DIGEST not in message


def test_ollama_judge_builds_the_adapter_and_records_identity() -> None:
    environ = {"TYPEVET_WORDING_JUDGE_PROVIDER": "ollama"}
    with live._ollama_judge(
        environ, adapter=_RecordingAdapter, get=_ollama_get()
    ) as judge:
        assert isinstance(judge.port, _RecordingAdapter)
        assert not judge.port.closed
    assert judge.port.closed
    (kwargs,) = _RecordingAdapter.built
    assert kwargs["base_url"] == BASE
    assert kwargs["default_model"] == "nimble"
    assert kwargs["api_key"] == live.OLLAMA_PLACEHOLDER_KEY
    assert "spend_cap" not in kwargs
    assert judge.model == "nimble"
    assert judge.spend is None
    assert judge.stoppers == ()
    assert judge.facts == {
        "base_url": BASE,
        "served_template": None,
        "ollama_version": "0.35.0",
        "model_digest": DIGEST,
    }


def test_ollama_judge_reads_base_and_model_knobs() -> None:
    base = "http://gpu-box:11434"
    environ = {
        "TYPEVET_OLLAMA_BASE": base + "/",
        "TYPEVET_WORDING_JUDGE": "other:7b",
    }
    with live._ollama_judge(
        environ, adapter=_RecordingAdapter, get=_ollama_get(base)
    ) as judge:
        assert judge.model == "other:7b"
    (kwargs,) = _RecordingAdapter.built
    assert kwargs["base_url"] == base
    assert kwargs["default_model"] == "other:7b"
    assert judge.facts["model_digest"] == "f" * 64


def test_ollama_judge_refuses_before_building_the_adapter() -> None:
    environ = {"TYPEVET_WORDING_JUDGE": "absent"}
    with (
        pytest.raises(ValueError, match="not listed"),
        live._ollama_judge(environ, adapter=_RecordingAdapter, get=_ollama_get()),
    ):
        pass
    assert _RecordingAdapter.built == []


def test_ollama_judge_needs_no_jev_key_or_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("JEV_API__KEY", "TYPESAFE_API_KEY", "JEV_API__SPEND_MAX_ATTEMPTS"):
        monkeypatch.delenv(name, raising=False)
    with live._ollama_judge({}, adapter=_RecordingAdapter, get=_ollama_get()) as judge:
        assert judge.model == "nimble"
