"""Offline semantic acceptance for saved CORD combined receipts ([#184][i184]).

Applies the frozen [#161][i161] revision 1 floors to routed combined outcomes.
Incomplete claim coverage, duplicate ids and non-finite measures reject acceptance.

Examples:
    ```python
    import json
    from pathlib import Path

    from typevet_evals.cord.semantic_acceptance import accept_combined_receipt

    receipt = json.loads(
        Path(
            "tests/fixtures/cord/semantic_acceptance/labeled_synthetic_pass.json"
        ).read_text()
    )
    outcome = accept_combined_receipt(receipt)
    assert outcome.accepted
    ```

See Also:
    - [typevet_evals.cord.semantic_metrics][]: shared confusion metrics
    - [typevet_evals.datasets.cord_expense][]: routing from judge labels

[i161]: https://github.com/Alberto-Codes/typevet/issues/161
[i184]: https://github.com/Alberto-Codes/typevet/issues/184
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from math import isfinite
from typing import Any, Final

from typevet_evals.cord.semantic_metrics import semantic_metrics
from typevet_evals.datasets.cord_expense import (
    CONTRADICTED,
    INSUFFICIENT,
    LABEL_ORDER,
    SUPPORTED,
    VERDICTS,
    route,
)

ANSWERABLE_ACCURACY_FLOOR: Final[float] = 0.67
CONTRADICTED_RECALL_FLOOR: Final[float] = 0.5
FALSE_CLEAR_CEILING: Final[float] = 0.25
ABSTENTION_FLOOR: Final[float] = 0.5
LABEL_SHARE_CEILING: Final[float] = 0.8
FROZEN_CLAIM_COUNT: Final[int] = 18

ACCURACY_CHECK: Final[str] = "answerable_accuracy"
CONTRADICTED_RECALL_CHECK: Final[str] = "contradicted_recall"
FALSE_CLEAR_CHECK: Final[str] = "false_clear_rate"
ABSTENTION_CHECK: Final[str] = "insufficient_abstention_rate"
LABEL_SHARE_CHECK: Final[str] = "max_label_share"

_JUDGE_LABELS: Final[frozenset[str]] = frozenset(LABEL_ORDER)
_VERDICTS: Final[frozenset[str]] = frozenset(VERDICTS)


class Bound(Enum):
    """Whether a threshold is a floor or a ceiling.

    Examples:
        ```python
        from typevet_evals.cord.semantic_acceptance import Bound

        assert Bound.FLOOR.value == "floor"
        ```
    """

    FLOOR = "floor"
    CEILING = "ceiling"


class CheckStatus(Enum):
    """Outcome of one threshold comparison.

    Examples:
        ```python
        from typevet_evals.cord.semantic_acceptance import CheckStatus

        assert CheckStatus.PASS is CheckStatus.PASS
        ```
    """

    PASS = auto()
    FAIL = auto()
    NOT_COMPUTABLE = auto()


@dataclass(frozen=True, slots=True)
class ThresholdCheck:
    """One measured value against a pinned floor or ceiling.

    Attributes:
        name (str): Check id, for example ``answerable_accuracy``.
        bound (Bound): Floor or ceiling comparison.
        limit (float): Pinned threshold from #161 revision 1.
        measured (float | None): Observed rate, or ``None`` when undefined.
        denominator (int): Gold count for this rate.

    Examples:
        ```python
        from typevet_evals.cord.semantic_acceptance import (
            Bound,
            ThresholdCheck,
        )

        ThresholdCheck(
            name="answerable_accuracy",
            bound=Bound.FLOOR,
            limit=0.67,
            measured=0.75,
            denominator=12,
        )
        ```
    """

    name: str
    bound: Bound
    limit: float
    measured: float | None
    denominator: int

    @property
    def status(self) -> CheckStatus:
        """Return pass, fail, or not computable for this check."""
        if self.measured is None:
            return CheckStatus.NOT_COMPUTABLE
        if not isfinite(self.measured):
            return CheckStatus.FAIL
        if self.bound is Bound.FLOOR:
            return CheckStatus.PASS if self.measured >= self.limit else CheckStatus.FAIL
        return CheckStatus.PASS if self.measured <= self.limit else CheckStatus.FAIL

    @property
    def passed(self) -> bool:
        """Return true only when the check status is pass."""
        return self.status is CheckStatus.PASS


@dataclass(frozen=True, slots=True)
class SemanticAcceptanceOutcome:
    """Acceptance verdict and per-floor diagnostics for one receipt.

    Attributes:
        accepted (bool): True when every check passed.
        claims (int): Claim count scored.
        checks (tuple[ThresholdCheck, ...]): One row per #161 floor.
        failures (tuple[str, ...]): Human-readable miss reasons.

    Examples:
        ```python
        from typevet_evals.cord.semantic_acceptance import (
            accept_semantic_outcome,
        )

        outcome = accept_semantic_outcome(
            {"R01-C1": "supported"},
            {"R01-C1": "supported"},
        )
        assert outcome.claims == 1
        ```
    """

    accepted: bool
    claims: int
    checks: tuple[ThresholdCheck, ...]
    failures: tuple[str, ...]

    def check(self, name: str) -> ThresholdCheck:
        """Return the check with ``name``.

        Args:
            name: Check id such as ``answerable_accuracy``.

        Returns:
            The matching ``ThresholdCheck``.

        Raises:
            KeyError: When no check uses ``name``.
        """
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)


def accept_semantic_outcome(
    gold: Mapping[str, str], predicted: Mapping[str, str]
) -> SemanticAcceptanceOutcome:
    """Score gold and predicted verdict maps against the #161 revision 1 floors.

    Args:
        gold: Gold verdict per claim id.
        predicted: Routed verdict per claim id.

    Returns:
        Acceptance outcome with one check per floor.

    Raises:
        ValueError: When the maps differ in claim ids or hold no claims.
    """
    if not gold:
        msg = "semantic outcome has no claim ids"
        raise ValueError(msg)
    if set(gold.keys()) != set(predicted.keys()):
        msg = "gold and predicted claim ids differ"
        raise ValueError(msg)
    metrics = semantic_metrics(gold, predicted)
    return _outcome_from_metrics(metrics, claims=len(gold))


def accept_combined_receipt(
    receipt: Mapping[str, Any],
    *,
    claim_ids: Sequence[str] | None = None,
) -> SemanticAcceptanceOutcome:
    """Accept or reject one saved combined receipt.

    Args:
        receipt: Receipt with ``cases`` and ``combined`` label rows.
        claim_ids: When set, case ids must match this frozen manifest order.

    Returns:
        Acceptance outcome for the combined arm.

    Raises:
        ValueError: When coverage, labels or gold verdicts are invalid.
    """
    combined = _require_mapping(receipt.get("combined"), "combined outcomes")
    cases = _require_case_list(receipt.get("cases"))
    gold, ids = _parse_cases(cases)
    if claim_ids is not None and tuple(claim_ids) != ids:
        msg = "claim ids do not match the frozen manifest"
        raise ValueError(msg)
    predicted = _parse_combined(combined, ids)
    return accept_semantic_outcome(gold, predicted)


def _require_mapping(raw: Any, where: str) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    msg = f"receipt is missing {where}"
    raise ValueError(msg)


def _require_case_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    msg = "receipt is missing cases"
    raise ValueError(msg)


def _parse_cases(cases: list[Any]) -> tuple[dict[str, str], tuple[str, ...]]:
    seen: list[str] = []
    gold: dict[str, str] = {}
    for row in cases:
        case = _require_mapping(row, "case row")
        claim_id = case.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            msg = "case is missing claim_id"
            raise ValueError(msg)
        if claim_id in gold:
            msg = f"duplicate claim_id {claim_id!r} in cases"
            raise ValueError(msg)
        verdict = case.get("expected_verdict")
        if not isinstance(verdict, str) or verdict not in _VERDICTS:
            msg = f"case {claim_id!r} has invalid expected_verdict"
            raise ValueError(msg)
        seen.append(claim_id)
        gold[claim_id] = verdict
    count = len(seen)
    if count != FROZEN_CLAIM_COUNT:
        msg = f"expected {FROZEN_CLAIM_COUNT} claims, got {count}"
        raise ValueError(msg)
    return gold, tuple(seen)


def _parse_combined(
    combined: Mapping[str, Any], claim_ids: tuple[str, ...]
) -> dict[str, str]:
    extra = set(combined.keys()) - set(claim_ids)
    if extra:
        cid = min(extra)
        msg = f"combined holds unexpected claim_id {cid!r}"
        raise ValueError(msg)
    predicted: dict[str, str] = {}
    for claim_id in claim_ids:
        row = _require_mapping(
            combined.get(claim_id), f"combined claim_id {claim_id!r}"
        )
        label = row.get("label")
        if not isinstance(label, str) or not label:
            msg = f"combined row {claim_id!r} has invalid label"
            raise ValueError(msg)
        predicted[claim_id] = _route_label(label, claim_id)
    return predicted


def _route_label(label: str, claim_id: str) -> str:
    if label in _JUDGE_LABELS:
        return route(label)
    if label in _VERDICTS:
        return label
    msg = f"combined row {claim_id!r} has invalid label {label!r}"
    raise ValueError(msg)


def _outcome_from_metrics(
    metrics: dict[str, object], *, claims: int
) -> SemanticAcceptanceOutcome:
    recall, predicted, false_supported = _metrics_parts(metrics)
    checks = (
        _accuracy_check(metrics),
        _contradicted_recall_check(recall, metrics),
        _false_clear_check(false_supported, metrics),
        _abstention_check(recall, metrics),
        _label_share_check(predicted, claims),
    )
    failures = tuple(
        _failure_reason(check)
        for check in checks
        if check.status is not CheckStatus.PASS
    )
    return SemanticAcceptanceOutcome(
        accepted=not failures,
        claims=claims,
        checks=checks,
        failures=failures,
    )


def _metrics_parts(
    metrics: dict[str, object],
) -> tuple[dict[str, float], dict[str, int], int]:
    recall = metrics.get("recall")
    predicted = metrics.get("predicted_counts")
    false_supported = metrics.get("false_supported")
    if (
        isinstance(recall, dict)
        and isinstance(predicted, dict)
        and isinstance(false_supported, int)
    ):
        return recall, predicted, false_supported
    msg = "semantic metrics missing recall, predicted_counts or false_supported"
    raise TypeError(msg)


def _accuracy_check(metrics: dict[str, object]) -> ThresholdCheck:
    answerable_n = _answerable_denominator(metrics)
    measured = None
    if answerable_n:
        measured = _answerable_correct(metrics) / answerable_n
    return ThresholdCheck(
        name=ACCURACY_CHECK,
        bound=Bound.FLOOR,
        limit=ANSWERABLE_ACCURACY_FLOOR,
        measured=measured,
        denominator=answerable_n,
    )


def _contradicted_recall_check(
    recall: dict[str, float], metrics: dict[str, object]
) -> ThresholdCheck:
    contradicted_n = _gold_count(metrics, CONTRADICTED)
    measured = float(recall[CONTRADICTED]) if contradicted_n else None
    return ThresholdCheck(
        name=CONTRADICTED_RECALL_CHECK,
        bound=Bound.FLOOR,
        limit=CONTRADICTED_RECALL_FLOOR,
        measured=measured,
        denominator=contradicted_n,
    )


def _false_clear_check(
    false_supported: int, metrics: dict[str, object]
) -> ThresholdCheck:
    non_supported = _non_supported_count(metrics)
    measured = false_supported / non_supported if non_supported else None
    return ThresholdCheck(
        name=FALSE_CLEAR_CHECK,
        bound=Bound.CEILING,
        limit=FALSE_CLEAR_CEILING,
        measured=measured,
        denominator=non_supported,
    )


def _abstention_check(
    recall: dict[str, float], metrics: dict[str, object]
) -> ThresholdCheck:
    insufficient_n = _gold_count(metrics, INSUFFICIENT)
    measured = float(recall[INSUFFICIENT]) if insufficient_n else None
    return ThresholdCheck(
        name=ABSTENTION_CHECK,
        bound=Bound.FLOOR,
        limit=ABSTENTION_FLOOR,
        measured=measured,
        denominator=insufficient_n,
    )


def _label_share_check(predicted: dict[str, int], claims: int) -> ThresholdCheck:
    measured = max(predicted.values()) / claims if predicted else None
    return ThresholdCheck(
        name=LABEL_SHARE_CHECK,
        bound=Bound.CEILING,
        limit=LABEL_SHARE_CEILING,
        measured=float(measured) if measured is not None else None,
        denominator=claims,
    )


def _failure_reason(check: ThresholdCheck) -> str:
    bound = check.bound.value
    if check.measured is None:
        return f"{check.name} is not computable"
    if check.bound is Bound.FLOOR:
        return f"{check.name} {check.measured} below {bound} {check.limit}"
    return f"{check.name} {check.measured} above {bound} {check.limit}"


def _answerable_correct(metrics: dict[str, object]) -> int:
    confusion = _confusion(metrics)
    correct = 0
    for verdict in (SUPPORTED, CONTRADICTED):
        row = confusion.get(verdict, {})
        correct += int(row.get(verdict, 0))
    return correct


def _answerable_denominator(metrics: dict[str, object]) -> int:
    confusion = _confusion(metrics)
    total = 0
    for verdict in (SUPPORTED, CONTRADICTED):
        row = confusion.get(verdict, {})
        total += sum(int(v) for v in row.values())
    return total


def _gold_count(metrics: dict[str, object], verdict: str) -> int:
    row = _confusion(metrics).get(verdict, {})
    return sum(int(v) for v in row.values())


def _non_supported_count(metrics: dict[str, object]) -> int:
    return _gold_count(metrics, CONTRADICTED) + _gold_count(metrics, INSUFFICIENT)


def _confusion(metrics: dict[str, object]) -> dict[str, dict[str, int]]:
    raw = metrics.get("confusion_gold_by_predicted")
    if isinstance(raw, dict):
        return raw
    msg = "semantic metrics missing confusion_gold_by_predicted"
    raise TypeError(msg)
