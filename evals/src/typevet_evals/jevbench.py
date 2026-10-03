"""Borrow a judgevet provider for the pinned upstream JevBench runner.

The adapter maps typed answers to exact labels. Upstream owns normalization,
scoring, timing, persistence and budget stop rules. No tariff is assumed.

Attributes:
    SystemOneAdapter (type): In-process adapter over a borrowed provider.

Examples:
    ```python
    from typevet_evals.jevbench import SystemOneAdapter

    adapter = SystemOneAdapter(provider, model="gemma")
    result = adapter.run(task)
    ```

See Also:
    - [typevet_evals.jevbench_run][]: Module command and session ownership.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from jevbench.adapters.base import DecisionResult
from judgevet import (
    ChoiceAnswer,
    JevError,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)
from judgevet.providers import ProviderError

if TYPE_CHECKING:
    from jevbench.tasks import Task
    from judgevet import SystemOnePort

__all__ = ["SystemOneAdapter"]


def _question(task: Task) -> dict[str, Any]:
    """Validate the label domain and copy only permitted question fields.

    Args:
        task: Upstream canonical task.

    Returns:
        The decision question without task metadata or gold.

    Raises:
        ValueError: The task has an unsupported or malformed label domain.
    """
    labels = task.labels
    if (
        not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("invalid_labels")
    kind = task.question.get("type")
    instructions = task.question.get("instructions")
    criteria = task.question.get("criteria")
    if not isinstance(instructions, str):
        raise TypeError("invalid_instructions")
    if kind == "noul":
        if labels != ["no", "yes"]:
            raise ValueError("invalid_noul_labels")
    elif kind == "choice":
        if not isinstance(criteria, dict) or set(criteria) != set(labels):
            raise ValueError("invalid_choice_criteria")
        if not all(isinstance(value, str) for value in criteria.values()):
            raise ValueError("invalid_choice_criteria")
    elif kind == "score":
        if not isinstance(criteria, list) or not all(
            isinstance(level, str) for level in criteria
        ):
            raise ValueError("invalid_score_criteria")
        if labels != [str(index) for index in range(len(criteria))]:
            raise ValueError("invalid_score_labels")
    else:
        raise ValueError("unsupported_question_type")
    question = {"type": kind, "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    return question


def _probability(value: Any) -> bool:
    """Check a finite numeric probability without coercion.

    Args:
        value: Provider value to check.

    Returns:
        Whether the value lies within the closed probability interval.
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 <= value <= 1
        and math.isfinite(value)
    )


def _distribution(response: SystemOneResponse, task: Task) -> dict[str, float]:
    """Map one matching typed answer without changing its probability mass.

    Args:
        response: Typed provider result.
        task: Validated task whose labels define the output domain.

    Returns:
        Probabilities over exact task labels.

    Raises:
        ValueError: The answer has the wrong type, labels or numeric values.
        TypeError: Response model or usage metadata has the wrong type.
    """
    if not isinstance(response.model, str) or not isinstance(response.usage, Usage):
        raise TypeError("invalid_response_metadata")
    answer = response.answers.get("decision")
    kind = task.question["type"]
    if kind == "noul" and isinstance(answer, NoulAnswer) and _probability(answer.noul):
        probs = {"no": 1 - answer.noul, "yes": answer.noul}
    elif kind == "choice" and isinstance(answer, ChoiceAnswer):
        if answer.choice not in task.labels or not _probability(answer.confidence):
            raise ValueError("invalid_choice")
        probs = dict(answer.probabilities)
    elif kind == "score" and isinstance(answer, ScoreAnswer):
        keys = [*answer.probabilities, *answer.legend]
        if any(not isinstance(key, int) or isinstance(key, bool) for key in keys):
            raise ValueError("invalid_score_keys")
        if answer.legend != dict(enumerate(task.question["criteria"])):
            raise ValueError("invalid_score_legend")
        if (
            isinstance(answer.score, bool)
            or not isinstance(answer.score, (int, float))
            or not 0 <= answer.score <= len(task.labels) - 1
            or not _probability(answer.confidence)
        ):
            raise ValueError("invalid_score")
        probs = {str(key): value for key, value in answer.probabilities.items()}
    else:
        raise ValueError("invalid_answer_type")
    if set(probs) != set(task.labels) or not all(
        _probability(value) for value in probs.values()
    ):
        raise ValueError("invalid_distribution")
    return probs


class SystemOneAdapter:
    """Adapt a borrowed SystemOnePort to upstream DecisionResult.

    Attributes:
        name (str): Route name in upstream records.
        model (str): Requested model identifier.
        price_input_per_m (None): Unknown input tariff.
        price_output_per_m (None): Unknown output tariff.
        cost_basis (str): No measured monetary cost.

    Examples:
        ```python
        adapter = SystemOneAdapter(provider, model="gemma")
        assert adapter.reserve_estimate(task) is None
        ```
    """

    name = "typevet_system_one"
    price_input_per_m = None
    price_output_per_m = None
    cost_basis = "unknown"

    def __init__(self, port: SystemOnePort, *, model: str) -> None:
        """Borrow the provider without opening or closing it.

        Args:
            port: Configured synchronous judgevet provider.
            model: Requested model identifier.
        """
        self._port = port
        self.model = model

    def build_request(self, task: Task) -> dict[str, Any]:
        """Build and check the JSON-safe provider request.

        Args:
            task: Canonical task with private scoring metadata.

        Returns:
            Only state, model and the decision question.

        Raises:
            ValueError: The task is malformed or contains nonfinite values.
            TypeError: The state cannot be serialized as JSON.
        """
        body = {
            "state": task.state,
            "model": self.model,
            "questions": {"decision": _question(task)},
        }
        json.dumps(body, allow_nan=False)
        return body

    def reserve_estimate(self, task: Task) -> None:
        """Leave the reserve amount to the upstream runner.

        Args:
            task: Task that upstream will reserve before calling the provider.

        This route has no configured tariff.
        """
        return

    def run(self, task: Task) -> DecisionResult:
        """Return mapped evidence or a safe known-failure record.

        Args:
            task: Canonical upstream task.

        Returns:
            An actual upstream result with no invented HTTP status or price.
        """
        result = DecisionResult(
            self.name, False, probs_source="native", model=self.model
        )
        try:
            body = self.build_request(task)
        except (ValueError, TypeError, KeyError, AttributeError):
            result.error = "invalid_request"
            return result
        result.request_body = body
        try:
            response = self._port.system_one(**body)
        except ProviderError as exc:
            result.error = type(exc).__name__
            result.status = exc.status_code if isinstance(exc, JevError) else None
            return result
        except (ValueError, TypeError, KeyError):
            result.error = "provider_validation_error"
            return result
        if not isinstance(response, SystemOneResponse):
            result.error = "invalid_response"
            return result
        try:
            probs = _distribution(response, task)
            raw = asdict(response)
            # Round-trip both validates and snapshots mutable provider data.
            raw = json.loads(json.dumps(raw, allow_nan=False))
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            result.error = "invalid_response"
            return result
        result.model = response.model
        result.usage = asdict(response.usage)
        result.raw = raw
        result.probs = probs
        result.ok = True
        return result
