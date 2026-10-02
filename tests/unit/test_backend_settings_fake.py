"""Unit tests for the offline ``fake`` judgment backend (#388)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet.adapters import inbound
from typevet.adapters.inbound.backend_settings import (
    generation_adapter,
    load_backend,
    open_judgment,
)
from typevet.adapters.inbound.fake_backend import (
    FakeJudgmentPort,
    require_live_session,
)
from typevet.adapters.outbound.vllm.judgment_factory import VllmJudgmentSession
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_questions import Choice, Noul, Score

pytestmark = pytest.mark.unit

_VAR = "TYPEVET_FAKE__DISTRIBUTIONS"
_MARKER = "SENTINEL-content-7d2a"
_ROUTE = Choice(criteria={"billing": "Money", "technical": "Bugs", "other": "Else"})
_QUALITY = Score(criteria=["Poor", "Fair", "Good", "Great"])


def _write(tmp_path: Path, payload: Any) -> str:
    path = tmp_path / "distributions.json"
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _questions() -> dict[str, Any]:
    return {"billing": Noul(), "route": _ROUTE, "quality": _QUALITY}


def test_load_backend_reads_fake() -> None:
    assert load_backend({"TYPEVET_BACKEND": "fake"}) == "fake"
    assert load_backend({"TYPEVET_BACKEND": " fake "}) == "fake"


def test_load_backend_error_names_all_three_backends() -> None:
    with pytest.raises(ValueError, match="TYPEVET_BACKEND") as caught:
        load_backend({"TYPEVET_BACKEND": "ollama"})
    message = str(caught.value)
    for backend in ("llama_cpp", "vllm", "fake"):
        assert backend in message


def test_open_judgment_is_exported_from_inbound() -> None:
    assert "open_judgment" in inbound.__all__
    assert inbound.open_judgment is open_judgment


def test_generation_adapter_refuses_fake_backend() -> None:
    with pytest.raises(ValueError, match="fake backend has no generation adapter"):
        generation_adapter({"TYPEVET_BACKEND": "fake"})


def test_fake_session_defaults_to_uniform_distributions() -> None:
    with open_judgment({"TYPEVET_BACKEND": "fake"}) as session:
        assert type(session).__name__ == "FakeJudgmentSession"
        assert session.model == "fake"
        response = session.port.judge("text", _questions(), session.model)

    assert response.model == "fake"
    assert response.nouls["billing"].noul == pytest.approx(0.5)
    route = response.choices["route"]
    assert route.probabilities == pytest.approx(
        {"billing": 1 / 3, "technical": 1 / 3, "other": 1 / 3}
    )
    assert route.choice == "billing"
    quality = response.scores["quality"]
    assert quality.probabilities == pytest.approx({0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    assert quality.score == pytest.approx(1.5)


def test_empty_variable_means_uniform() -> None:
    environ = {"TYPEVET_BACKEND": "fake", _VAR: "  "}
    with open_judgment(environ) as session:
        response = session.port.judge("t", {"billing": Noul()}, session.model)
    assert response.nouls["billing"].noul == pytest.approx(0.5)


def test_fake_session_reads_file_distributions(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "billing": 0.9,
            "route": {"technical": 3, "billing": 1},
            "quality": {"2": 3.0, "0": 1.0},
            "unused": 0.1,
        },
    )
    questions = {**_questions(), "extra": Noul(), "level": Score(criteria=["a", "b"])}
    with open_judgment({"TYPEVET_BACKEND": "fake", _VAR: path}) as session:
        response = session.port.judge("text", questions, session.model)
        again = session.port.judge("text", {"billing": Noul()}, session.model)

    assert response.nouls["billing"].noul == pytest.approx(0.9)
    route = response.choices["route"]
    assert route.choice == "technical"
    assert route.probabilities == pytest.approx(
        {"billing": 0.25, "technical": 0.75, "other": 0.0}
    )
    quality = response.scores["quality"]
    assert quality.score == pytest.approx(1.5)
    assert quality.probabilities == pytest.approx({0: 0.25, 1: 0.0, 2: 0.75, 3: 0.0})
    assert response.nouls["extra"].noul == pytest.approx(0.5)
    assert response.scores["level"].probabilities == pytest.approx({0: 0.5, 1: 0.5})
    assert again.nouls["billing"].noul == pytest.approx(0.9)


@pytest.mark.parametrize(
    "payload",
    [
        f"not json {_MARKER}",
        [_MARKER],
        {_MARKER: "text"},
        {_MARKER: True},
        {_MARKER: {"label": "heavy"}},
        {_MARKER: {"label": False}},
        {_MARKER: None},
    ],
    ids=[
        "not-json",
        "not-object",
        "string",
        "bool",
        "string-weight",
        "bool-weight",
        "null",
    ],
)
def test_invalid_file_raises_value_error_naming_the_variable(
    tmp_path: Path, payload: Any
) -> None:
    path = _write(tmp_path, payload)
    with (
        pytest.raises(ValueError, match=_VAR) as caught,
        open_judgment({"TYPEVET_BACKEND": "fake", _VAR: path}),
    ):
        pass
    assert _MARKER not in str(caught.value)
    assert caught.value.__cause__ is None
    error = caught.value
    assert error.__context__ is None or error.__suppress_context__


def test_missing_file_raises_value_error_naming_the_variable(tmp_path: Path) -> None:
    path = str(tmp_path / f"{_MARKER}.json")
    with (
        pytest.raises(ValueError, match=_VAR) as caught,
        open_judgment({"TYPEVET_BACKEND": "fake", _VAR: path}),
    ):
        pass
    assert _MARKER not in str(caught.value)
    assert caught.value.__cause__ is None


def test_score_key_that_is_not_an_integer_raises_value_error(tmp_path: Path) -> None:
    path = _write(tmp_path, {"quality": {f"level-{_MARKER}": 1.0}})
    environ = {"TYPEVET_BACKEND": "fake", _VAR: path}
    with (
        open_judgment(environ) as session,
        pytest.raises(ValueError, match=_VAR) as caught,
    ):
        session.port.judge("t", {"quality": _QUALITY}, session.model)
    assert _MARKER not in str(caught.value)


def test_raw_wire_question_passes_to_the_fake_and_raises() -> None:
    raw = {"type": "noul", "instructions": "Is it?"}
    with (
        open_judgment({"TYPEVET_BACKEND": "fake"}) as session,
        pytest.raises(JudgmentValidationError, match="unsupported wire question"),
    ):
        session.port.judge("t", {"raw": raw}, session.model)


def test_require_live_session_passes_live_and_refuses_fake() -> None:
    with httpx.Client() as client:
        live = VllmJudgmentSession(port=FakeJudgmentPort({}), client=client, model="m")
        assert require_live_session(live) is live
    with (
        open_judgment({"TYPEVET_BACKEND": "fake"}) as session,
        pytest.raises(ValueError, match="TYPEVET_BACKEND") as caught,
    ):
        require_live_session(session)
    assert "fake" in str(caught.value)
