"""Offline raw probability evidence for PSAI vision Choice matrix ([#180][i180]).

Scratchpad receipts once stored ``sum(exp(lp - max(lp)))`` as ``raw_mass``.
That value is the softmax denominator over declared candidates only. Raw mass
is ``sum(exp(lp))`` over the same candidates. Normalized confidence is the
softmax quotient and is preserved when logprobs are unchanged.

Examples:
    ```python
    from typevet_evals.psai_vision_probability_evidence import (
        raw_candidate_mass,
        correct_matrix_receipt,
    )

    mass = raw_candidate_mass((-5.15, -6.92))
    assert mass < 0.01
    ```

See Also:
    - [docs.reference.psai-vision-choice-probability-evidence][]: reporting
    - [typevet_evals.datasets.psai_vision_controls][]: omitted non-credit

[i180]: https://github.com/Alberto-Codes/typevet/issues/180
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

ARTIFACT_VERSION: Final[str] = "vision_choice_probability_evidence_v1"
OMITTED_CONDITION: Final[str] = "omitted"
_PROB_SUM_TOLERANCE: Final[float] = 1e-6


def log_sum_exp(logprobs: Sequence[float]) -> float:
    """Stable log of ``sum(exp(logprobs))`` for finite inputs.

    Args:
        logprobs: Candidate logprobs in label order.

    Returns:
        Natural log of the unnormalized candidate mass.

    Raises:
        ValueError: When ``logprobs`` is empty or holds a non-finite value.
    """
    if not logprobs:
        msg = "logprobs must not be empty"
        raise ValueError(msg)
    peak = max(logprobs)
    if not math.isfinite(peak):
        msg = f"non-finite peak logprob: {peak!r}"
        raise ValueError(msg)
    total = sum(math.exp(lp - peak) for lp in logprobs)
    if total <= 0.0 or not math.isfinite(total):
        msg = "log-sum-exp produced non-finite mass"
        raise ValueError(msg)
    return peak + math.log(total)


def raw_candidate_mass(logprobs: Sequence[float]) -> float:
    """Return ``sum(exp(lp))`` over declared single-token candidates.

    Args:
        logprobs: Candidate logprobs in label order.

    Returns:
        Unnormalized probability mass for the scored candidate set.
    """
    return math.exp(log_sum_exp(logprobs))


def legacy_denominator_mass(logprobs: Sequence[float]) -> float:
    """Return the scratchpad bug value ``sum(exp(lp - max(lp)))``.

    Args:
        logprobs: Candidate logprobs in label order.

    Returns:
        Softmax denominator with peak removed (approximately one for two labels).
    """
    peak = max(logprobs)
    return sum(math.exp(lp - peak) for lp in logprobs)


def normalized_confidence(
    logprobs: Sequence[float],
    labels: Sequence[str],
) -> dict[str, float]:
    """Softmax normalized confidence over declared candidates only.

    Args:
        logprobs: Candidate logprobs aligned with ``labels``.
        labels: Candidate labels in scoring order.

    Returns:
        Label-keyed probabilities that sum to one within tolerance.

    Raises:
        ValueError: When lengths differ or normalization is non-finite.
    """
    if len(logprobs) != len(labels):
        msg = "logprobs and labels must have the same length"
        raise ValueError(msg)
    peak = max(logprobs)
    weights = [math.exp(lp - peak) for lp in logprobs]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        msg = "softmax denominator is not finite"
        raise ValueError(msg)
    probs = {
        label: weight / total for label, weight in zip(labels, weights, strict=True)
    }
    prob_sum = sum(probs.values())
    if abs(prob_sum - 1.0) > _PROB_SUM_TOLERANCE:
        msg = f"normalized probabilities sum to {prob_sum!r}"
        raise ValueError(msg)
    return probs


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-serializable mapping with stable key order.

    Args:
        payload: Receipt or artifact body.

    Returns:
        Lowercase hex SHA-256 of canonical JSON bytes.
    """
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _sanitize_outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(outcome)
    cleaned.pop("prefix_tail", None)
    return cleaned


def sanitize_source_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Drop long prefix tails while keeping scoring evidence fields.

    Args:
        receipt: Full live or scratchpad matrix receipt.

    Returns:
        Sanitized receipt suitable for tracked fixtures.
    """
    copy = dict(receipt)
    outcomes = receipt.get("outcomes")
    if isinstance(outcomes, list):
        copy["outcomes"] = [_sanitize_outcome(row) for row in outcomes]
    return copy


def correct_outcome_mass(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute mass fields from preserved ``raw_logprobs``.

    Args:
        outcome: One matrix outcome with ``candidate_labels`` and logprobs.

    Returns:
        Outcome copy with ``raw_mass``, ``legacy_denominator_mass``,
        ``normalized_confidence_recomputed``, and ``recorded_raw_mass``.

    Raises:
        TypeError: When label or logprob fields are not lists.
    """
    labels_raw = outcome.get("candidate_labels")
    logprobs_raw = outcome.get("raw_logprobs")
    if not isinstance(labels_raw, list) or not isinstance(logprobs_raw, list):
        msg = "outcome missing candidate_labels or raw_logprobs lists"
        raise TypeError(msg)
    labels = tuple(str(label) for label in labels_raw)
    logprobs = tuple(float(lp) for lp in logprobs_raw)
    corrected = _sanitize_outcome(outcome)
    corrected["recorded_raw_mass"] = outcome.get("raw_mass")
    corrected["legacy_denominator_mass"] = legacy_denominator_mass(logprobs)
    corrected["raw_mass"] = raw_candidate_mass(logprobs)
    corrected["normalized_confidence_recomputed"] = normalized_confidence(
        logprobs,
        labels,
    )
    return corrected


def _c10_repair_block(
    receipt: Mapping[str, Any],
    *,
    source_sha256: str | None,
) -> dict[str, Any]:
    repair_outcomes = receipt.get("outcomes")
    if not isinstance(repair_outcomes, list):
        msg = "C10 repair receipt missing outcomes list"
        raise TypeError(msg)
    return {
        "source_sha256": source_sha256,
        "outcomes": [correct_outcome_mass(row) for row in repair_outcomes],
        "swap_donor": receipt.get("c10_swap_donor"),
        "note": (
            "Protocol repair: C10 swap donor re-pinned to opposite-gold C04 "
            "after neg/neg f18gc2 pin failed semantic swap."
        ),
    }


def _reporting_envelope(
    matrix_receipt: Mapping[str, Any],
    *,
    matrix_source_sha256: str,
    c10_repair_source_sha256: str | None,
    omitted_call_ids: list[str],
    has_c10_repair: bool,
) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "issue": 180,
        "parent_issue": 109,
        "sources": {
            "matrix_receipt_sha256": matrix_source_sha256,
            "c10_repair_receipt_sha256": c10_repair_source_sha256,
        },
        "model_id": matrix_receipt.get("model_id"),
        "served_template": matrix_receipt.get("served_template"),
        "baseline_commit": matrix_receipt.get("baseline_commit"),
        "prior_invalid_hub_matrix": {
            "status": matrix_receipt.get("prior_matrix_status"),
            "note": (
                "180-matrix-post187-receipt.json used Hub category on task_name "
                "(protocol B). Void for rev2; not debited against rev2 cap."
            ),
        },
        "c10_donor_correction": {
            "initial_matrix_swap_donor_c10": "cmcc8u6ym018l1p1yxhf18gc2",
            "repaired_swap_donor_c10": "cmcc8u6yd00wr1p1yj7aot3ae",
            "repair_scoring_calls": 2 if has_c10_repair else 0,
        },
        "cumulative_scoring_calls": {
            "rev2_visual_choice_matrix": 16,
            "invalid_post187_hub_matrix": 16,
            "c10_donor_repair": 2 if has_c10_repair else 0,
            "note": (
                "Invalid Hub matrix and rev2 matrix are separate runs; "
                "repair is additive."
            ),
        },
        "omitted_arms": {
            "count": len(omitted_call_ids),
            "semantic_credit": False,
            "call_ids": omitted_call_ids,
            "note": "Four omitted visibility arms are expected non-credits.",
        },
        "completeness_verdict": (
            "Semantic evidence is complete for rev2 visual Choice plus C10 repair "
            "when sources match pinned SHA-256 digests. This is not a pristine "
            "16/16 semantic pass: four omitted arms fail by design, the initial "
            "matrix had one swap fail before donor repair, and the earlier Hub "
            "matrix remains invalid."
        ),
        "summary": matrix_receipt.get("summary"),
        "budget": matrix_receipt.get("budget"),
    }


def build_corrected_artifact(
    *,
    matrix_receipt: Mapping[str, Any],
    matrix_source_sha256: str,
    c10_repair_receipt: Mapping[str, Any] | None = None,
    c10_repair_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish versioned corrected evidence linked to source hashes.

    Args:
        matrix_receipt: Sanitized rev2 visual Choice matrix (16 calls).
        matrix_source_sha256: Digest of ``source_matrix_v1.json`` bytes.
        c10_repair_receipt: Optional two-call C10 donor repair receipt.
        c10_repair_source_sha256: Digest of repair source fixture when set.

    Returns:
        Corrected artifact with budget reconciliation and completeness verdict.

    Raises:
        TypeError: When receipt outcome lists are missing or wrong type.
    """
    matrix_outcomes = matrix_receipt.get("outcomes")
    if not isinstance(matrix_outcomes, list):
        msg = "matrix receipt missing outcomes list"
        raise TypeError(msg)
    corrected_matrix = [correct_outcome_mass(row) for row in matrix_outcomes]
    omitted_call_ids = [
        str(row["call_id"])
        for row in corrected_matrix
        if row.get("condition") == OMITTED_CONDITION
    ]
    repair_block = None
    if c10_repair_receipt is not None:
        repair_block = _c10_repair_block(
            c10_repair_receipt,
            source_sha256=c10_repair_source_sha256,
        )
    envelope = _reporting_envelope(
        matrix_receipt,
        matrix_source_sha256=matrix_source_sha256,
        c10_repair_source_sha256=c10_repair_source_sha256,
        omitted_call_ids=omitted_call_ids,
        has_c10_repair=c10_repair_receipt is not None,
    )
    envelope["matrix_outcomes"] = corrected_matrix
    envelope["c10_repair"] = repair_block
    return envelope


def correct_matrix_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Correct every outcome in one matrix receipt mapping.

    Args:
        receipt: Matrix receipt with an ``outcomes`` list.

    Returns:
        Receipt copy whose outcomes carry repaired ``raw_mass`` values.

    Raises:
        TypeError: When the receipt has no outcomes list.
    """
    copy = dict(receipt)
    outcomes = receipt.get("outcomes")
    if not isinstance(outcomes, list):
        msg = "receipt missing outcomes list"
        raise TypeError(msg)
    copy["outcomes"] = [correct_outcome_mass(row) for row in outcomes]
    return copy
