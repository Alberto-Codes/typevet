"""Validate and snapshot Noul, Choice and Score requests before judgment IO.

See Also:
    - [typevet_consumer_bridge.adapter][]: Synchronous entry point.


Examples:
    ```python
    from typevet_consumer_bridge import BridgeSettings

    settings = BridgeSettings("http://localhost:8080", 30.0, "served-model")
    ```
"""

from collections.abc import Mapping
from math import isfinite
from typing import Any

from judgevet import Choice, Noul, Score

from typevet.domain import Choice as EngineChoice
from typevet.domain import Noul as EngineNoul
from typevet.domain import Score as EngineScore
from typevet_consumer_bridge.errors import BridgeRequestError

_MIN_OPTIONS = 2
_MAX_OPTIONS = 24


def _snapshot(value: object, active: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    if not isinstance(value, (dict, list)) or id(value) in active:
        raise BridgeRequestError("Invalid JSON state.")
    active.add(id(value))
    try:
        if isinstance(value, list):
            return [_snapshot(item, active) for item in value]
        if any(not isinstance(key, str) for key in value):
            raise BridgeRequestError("Invalid JSON state.")
        return {key: _snapshot(item, active) for key, item in value.items()}
    finally:
        active.remove(id(value))


def snapshot_state(state: object) -> Any:
    """Return a validated deep snapshot of text, object or array state.

    Args:
        state: Caller-owned state.

    Returns:
        A fresh JSON container or the exact text.

    Raises:
        BridgeRequestError: If state is not supported JSON.
    """
    if not isinstance(state, (str, dict, list)):
        raise BridgeRequestError("Invalid JSON state.")
    return _snapshot(state, set())


def _convert(question: object) -> EngineNoul | EngineChoice | EngineScore:
    if isinstance(question, (Noul, Choice, Score)):
        kind = (
            "noul"
            if isinstance(question, Noul)
            else ("choice" if isinstance(question, Choice) else "score")
        )
        instructions, criteria = question.instructions, question.criteria
    elif isinstance(question, Mapping):
        kind = question.get("type")
        if kind not in ("noul", "choice", "score") or set(question) - {
            "type",
            "instructions",
            "criteria",
        }:
            raise BridgeRequestError("Unsupported question form.")
        instructions, criteria = question.get("instructions"), question.get("criteria")
    else:
        raise BridgeRequestError("Unsupported question form.")
    if instructions is not None and not isinstance(instructions, str):
        raise BridgeRequestError("Invalid question instructions.")
    if kind == "choice":
        if (
            not isinstance(criteria, Mapping)
            or not _MIN_OPTIONS <= len(criteria) <= _MAX_OPTIONS
            or any(
                not isinstance(key, str)
                or not key.strip()
                or (value is not None and not isinstance(value, str))
                for key, value in criteria.items()
            )
        ):
            raise BridgeRequestError("Invalid Choice criteria.")
        return EngineChoice(instructions=instructions, criteria=dict(criteria))
    if kind == "score":
        if (
            not isinstance(criteria, (list, tuple))
            or not _MIN_OPTIONS <= len(criteria) <= _MAX_OPTIONS
            or any(not isinstance(value, str) for value in criteria)
        ):
            raise BridgeRequestError("Invalid Score criteria.")
        return EngineScore(instructions=instructions, criteria=list(criteria))
    if criteria is not None:
        if not isinstance(criteria, Mapping) or any(
            key not in {"true", "false"} or not isinstance(value, str)
            for key, value in criteria.items()
        ):
            raise BridgeRequestError("Invalid Noul criteria.")
        criteria = dict(criteria)
    return EngineNoul(instructions=instructions, criteria=criteria)


def convert_questions(
    questions: object,
) -> dict[str, EngineNoul | EngineChoice | EngineScore]:
    """Reconstruct engine Noul, Choice and Score from validated consumer forms.

    Args:
        questions: Nonempty mapping of exact IDs to questions.

    Returns:
        Fresh engine questions in caller order.

    Raises:
        BridgeRequestError: If any question is invalid or unsupported.
    """
    if not isinstance(questions, Mapping) or not questions:
        raise BridgeRequestError("Invalid question mapping.")
    converted = {}
    for key, question in questions.items():
        if not isinstance(key, str) or not key.strip():
            raise BridgeRequestError("Invalid question ID.")
        converted[key] = _convert(question)
    return converted
