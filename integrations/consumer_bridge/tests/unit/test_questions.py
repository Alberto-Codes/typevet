"""Verify whole-request validation and exact consumer conversion.

Examples:
    ```bash
    uv run pytest -q integrations/consumer_bridge/tests/unit/test_questions.py
    ```

See Also:
    - [typevet_consumer_bridge.adapter][]: Public conversion and ownership.
"""

import pytest
from judgevet import Choice, Noul, Score
from typevet_consumer_bridge import BridgeRequestError
from typevet_consumer_bridge.questions import convert_questions, snapshot_state

from typevet.domain import Choice as EngineChoice
from typevet.domain import Noul as EngineNoul
from typevet.domain import Score as EngineScore

pytestmark = pytest.mark.unit


def require(condition: bool, message: str) -> None:
    """Fail a named contract assertion.

    Raises:
        AssertionError: If the named contract assertion fails.
    """
    if not condition:
        raise AssertionError(message)


def test_noul_reconstruction() -> None:
    """Preserve exact IDs, instructions, criteria and insertion order."""
    instructions = "  Check total?\n"
    criteria = {"false": "wrong", "true": "exact"}
    questions = {" Total ": Noul(instructions=instructions, criteria=criteria)}
    result = convert_questions(questions)
    require(list(result) == [" Total "], "exact ID")
    converted = result[" Total "]
    require(isinstance(converted, EngineNoul), "engine-owned input")
    require(converted.instructions == instructions, "caller instructions")
    require(converted.criteria == criteria, "caller criteria")
    require(converted.criteria is not criteria, "criteria snapshot")


@pytest.mark.parametrize("criteria", [None, {}, {"true": "yes"}])
def test_raw_noul(criteria: object) -> None:
    """Raw and object forms use the same validated conversion."""
    converted = convert_questions({"q": {"type": "noul", "criteria": criteria}})
    require(converted["q"].criteria == criteria, "raw criteria preserved")
    require(converted["q"].instructions is None, "None instructions")


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        {"": Noul()},
        {1: Noul()},
        {"q": EngineNoul()},
        {"q": {"type": "NOUL"}},
        {"q": {}},
        {"q": {"type": "noul", "extra": 1}},
        {"q": Noul(instructions={"text": "bad"})},
        {"q": Noul(criteria={"TRUE": "bad"})},
        {"q": Noul(criteria={"true": None})},
    ],
)
def test_invalid_questions(invalid: object) -> None:
    """Reject unsupported shapes without repr coercion."""
    with pytest.raises(BridgeRequestError):
        convert_questions(invalid)


def test_state_snapshot() -> None:
    """Deep-copy JSON state while preserving its order and values."""
    state = {"b": [1, None, True, {"z": "  exact\n"}], "a": 2.5}
    snapshot = snapshot_state(state)
    require(snapshot == state and snapshot is not state, "deep snapshot")
    state["b"].append(None)
    require(snapshot != state, "nested snapshot")
    require(snapshot_state("  exact\n") == "  exact\n", "exact text")


@pytest.mark.parametrize(
    "state",
    [
        None,
        1,
        True,
        {1: "bad"},
        {"x": object()},
        [float("nan")],
        [float("inf")],
        (1, 2),
    ],
)
def test_invalid_state(state: object) -> None:
    """Reject non-JSON states and nonfinite values."""
    with pytest.raises(BridgeRequestError):
        snapshot_state(state)


def test_cyclic_state() -> None:
    """Reject cycles without recursive repr or partial IO."""
    state: list[object] = []
    state.append(state)
    with pytest.raises(BridgeRequestError):
        snapshot_state(state)


@pytest.mark.parametrize("count", [2, 24])
@pytest.mark.parametrize("kind", ["choice", "score"])
def test_choice_score_forms(count: int, kind: str) -> None:
    """Preserve ordered criteria and instructions across raw and object forms."""
    instructions = "  Exact caller instruction\n"
    criteria = {f" label {i} ": None if i == 0 else str(i) for i in range(count)}
    levels = tuple(str(i) for i in range(count))
    question = (
        Choice(criteria=criteria, instructions=instructions)
        if kind == "choice"
        else Score(criteria=levels, instructions=instructions)
    )
    expected = criteria if kind == "choice" else list(levels)
    raw = {
        "type": kind,
        "criteria": criteria if kind == "choice" else levels,
        "instructions": instructions,
    }
    results = convert_questions({" object ": question, "raw": raw})
    require(list(results) == [" object ", "raw"], "ordered exact IDs")
    for result in results.values():
        require(
            type(result) is (EngineChoice if kind == "choice" else EngineScore),
            "explicit engine reconstruction",
        )
        require(result.instructions == instructions, "exact instructions")
        require(result.criteria == expected, "exact ordered criteria")
        require(result.criteria is not question.criteria, "fresh criteria")
        if isinstance(result, EngineChoice):
            require(list(result.criteria) == list(criteria), "label insertion order")


@pytest.mark.parametrize(
    "criteria",
    [
        None,
        {},
        {"a": "A"},
        {str(i): None for i in range(25)},
        {" ": None, "b": "B"},
        {1: "A", "b": "B"},
        {"a": {}, "b": "B"},
        {"a": [], "b": "B"},
    ],
)
def test_invalid_choice(criteria: object) -> None:
    """Reject unsupported Choice option counts, labels and descriptions."""
    with pytest.raises(BridgeRequestError):
        convert_questions({"q": {"type": "choice", "criteria": criteria}})


@pytest.mark.parametrize(
    "criteria",
    [
        None,
        [],
        ["a"],
        ["a"] * 25,
        "ab",
        {0: "a", 1: "b"},
        [None, "b"],
        [{}, "b"],
        [[], "b"],
    ],
)
def test_invalid_score(criteria: object) -> None:
    """Reject unsupported Score levels without coercion."""
    with pytest.raises(BridgeRequestError):
        convert_questions({"q": {"type": "score", "criteria": criteria}})
