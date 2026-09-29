"""Structural fail-closed checks for consumer receipts ([#177][i177]).

Examples:
    ```python
    from typevet_evals.psai_vision_consumer.receipt_structure import (
        protocol_structural_failures,
    )

    assert protocol_structural_failures({}) == ["receipt is empty"]
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.protocol][]: frozen pins

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from typevet_evals.psai_vision_consumer.accounting import (
    FROZEN_CONSUMER_CASE_UIDS,
    TEXT_ANNOTATION_JUDGE_UIDS,
)
from typevet_evals.psai_vision_consumer.protocol import (
    FROZEN_JUDGMENT_CALLS,
    FROZEN_PROTOCOL_REVISION,
    FROZEN_SCORING_REQUESTS,
    expected_matrix_row_keys,
    expected_paired_ordering_uids,
)


def _finite_prob(value: object) -> bool:
    return isinstance(value, (int, float)) and isfinite(float(value))


def _noul_answer_failures(name: str, payload: Mapping[str, Any]) -> list[str]:
    prob = payload.get("noul")
    if not isinstance(prob, (int, float)) or not isfinite(float(prob)):
        return [f"answer {name!r} has non-finite noul"]
    prob_float = float(prob)
    if not 0.0 <= prob_float <= 1.0:
        return [f"answer {name!r} noul out of [0,1]"]
    return []


def _mapping_prob_failures(
    name: str,
    probs: object,
    *,
    label: str,
) -> list[str]:
    if not isinstance(probs, Mapping):
        return []
    failures: list[str] = []
    for key, value in probs.items():
        if not _finite_prob(value):
            failures.append(f"answer {name!r} {label} prob {key!r} non-finite")
    return failures


def _answer_probability_failures(answers: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, payload in answers.items():
        if not isinstance(payload, Mapping):
            failures.append(f"answer {name!r} is not an object")
            continue
        kind = payload.get("kind")
        if kind == "Noul":
            failures.extend(_noul_answer_failures(name, payload))
        elif kind == "Choice":
            failures.extend(
                _mapping_prob_failures(
                    name, payload.get("probabilities"), label="choice"
                )
            )
        elif kind == "Score":
            failures.extend(
                _mapping_prob_failures(
                    name, payload.get("probabilities"), label="score"
                )
            )
    return failures


def _revision_and_count_failures(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    revision = receipt.get("consumer_live_protocol_revision")
    if revision != FROZEN_PROTOCOL_REVISION:
        failures.append(
            f"consumer_live_protocol_revision {revision!r} != {FROZEN_PROTOCOL_REVISION}"
        )
    judgment = receipt.get("judgment_call_count")
    scoring = receipt.get("scoring_request_count")
    if not isinstance(judgment, int) or judgment <= 0:
        failures.append("judgment_call_count missing or zero")
    if not isinstance(scoring, int) or scoring <= 0:
        failures.append("scoring_request_count missing or zero")
    if judgment != FROZEN_JUDGMENT_CALLS:
        failures.append(
            f"judgment_call_count {judgment!r} != frozen {FROZEN_JUDGMENT_CALLS}"
        )
    if scoring != FROZEN_SCORING_REQUESTS:
        failures.append(
            f"scoring_request_count {scoring!r} != frozen {FROZEN_SCORING_REQUESTS}"
        )
    require_live = receipt.get("require_live") is True
    aux = receipt.get("auxiliary_http_count")
    if require_live and (not isinstance(aux, int) or aux <= 0):
        failures.append("auxiliary_http_count missing or zero for live receipt")
    return failures


def _capability_failures(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    require_live = receipt.get("require_live") is True
    capability = receipt.get("capability")
    if not isinstance(capability, Mapping):
        return ["missing capability block"]
    if require_live and capability.get("vision") is not True:
        failures.append("capability.vision is not true for live receipt")
    elif capability.get("vision") is False:
        failures.append("capability.vision is false")
    negative = receipt.get("unsupported_capability_negative")
    if not isinstance(negative, Mapping):
        failures.append("missing unsupported_capability_negative block")
    elif negative.get("ok") is not True:
        failures.append("unsupported_capability_negative did not pass")
    return failures


def _matrix_row_failures(matrix: list[Any]) -> list[str]:
    failures: list[str] = []
    expected_keys = expected_matrix_row_keys()
    seen_keys: set[tuple[str, str, str]] = set()
    for row in matrix:
        if not isinstance(row, Mapping):
            return ["matrix row is not an object"]
        uid = row.get("unique_data_id")
        condition = row.get("condition")
        leg = row.get("leg")
        if not isinstance(uid, str) or not isinstance(condition, str):
            return ["matrix row missing unique_data_id or condition"]
        if not isinstance(leg, str):
            return [f"matrix row {uid!r} missing leg"]
        key = (uid, condition, leg)
        if (
            uid not in FROZEN_CONSUMER_CASE_UIDS
            and uid not in TEXT_ANNOTATION_JUDGE_UIDS
        ):
            failures.append(f"matrix row unknown unique_data_id {uid!r}")
        if key not in expected_keys:
            failures.append(f"matrix row unexpected key {key!r}")
        if key in seen_keys:
            failures.append(f"duplicate matrix row {key!r}")
        seen_keys.add(key)
        answers = row.get("answers")
        if not isinstance(answers, Mapping) or not answers:
            failures.append(f"matrix row {key!r} missing answers map")
        else:
            failures.extend(_answer_probability_failures(answers))
    missing = expected_keys - seen_keys
    if missing:
        failures.append(f"missing matrix rows for keys {sorted(missing)!r}")
    return failures


def _matrix_failures(receipt: Mapping[str, Any]) -> list[str]:
    matrix = receipt.get("matrix_rows")
    if not isinstance(matrix, list) or not matrix:
        return ["matrix_rows missing or empty"]
    failures = _matrix_row_failures(matrix)
    judgment = receipt.get("judgment_call_count")
    if isinstance(judgment, int) and len(matrix) != judgment:
        failures.append(f"matrix_rows {len(matrix)} != judgment_call_count {judgment}")
    return failures


def _paired_ordering_failures(receipt: Mapping[str, Any]) -> list[str]:
    pairs = receipt.get("paired_ordering")
    if not isinstance(pairs, list) or not pairs:
        return ["paired_ordering missing or empty"]
    failures: list[str] = []
    pair_uids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            return ["paired_ordering entry is not an object"]
        uid = pair.get("unique_data_id")
        if not isinstance(uid, str):
            return ["paired_ordering entry missing unique_data_id"]
        if uid not in expected_paired_ordering_uids():
            failures.append(f"paired_ordering unknown unique_data_id {uid!r}")
        if uid in pair_uids:
            failures.append(f"duplicate paired_ordering uid {uid!r}")
        pair_uids.add(uid)
        if not pair.get("ordered"):
            failures.append(f"paired_ordering failed for row {uid!r}")
    missing_pairs = expected_paired_ordering_uids() - pair_uids
    if missing_pairs:
        failures.append(f"missing paired_ordering for {sorted(missing_pairs)!r}")
    return failures


def _observed_scoring_failures(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    scoring = receipt.get("scoring_request_count")
    observed = receipt.get("scoring_requests_observed")
    if isinstance(scoring, int) and isinstance(observed, int) and observed != scoring:
        failures.append(
            f"scoring_requests_observed {observed} != scoring_request_count {scoring}"
        )
    matrix = receipt.get("matrix_rows")
    if not isinstance(matrix, list) or not isinstance(observed, int):
        return failures
    answer_total = sum(
        len(row["answers"])
        for row in matrix
        if isinstance(row, Mapping) and isinstance(row.get("answers"), Mapping)
    )
    if answer_total != observed:
        failures.append(
            f"scoring_requests_observed {observed} != "
            f"serialized answer count {answer_total}"
        )
    return failures


def _pin_failures(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    manifest_pin = receipt.get("manifest_sha256")
    if not isinstance(manifest_pin, str) or not manifest_pin.strip():
        failures.append("manifest_sha256 pin missing")
    image_pins = receipt.get("frozen_case_image_digests")
    if not isinstance(image_pins, list) or len(image_pins) != len(
        FROZEN_CONSUMER_CASE_UIDS
    ):
        failures.append("frozen_case_image_digests pin missing or incomplete")
    return failures


def protocol_structural_failures(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed structural checks independent of machine-local paths.

    Args:
        receipt: Consumer proof receipt dict.

    Returns:
        Human-readable failure messages (empty when structure matches rev 2).
    """
    if not receipt:
        return ["receipt is empty"]
    failures: list[str] = []
    failures.extend(_revision_and_count_failures(receipt))
    failures.extend(_capability_failures(receipt))
    failures.extend(_matrix_failures(receipt))
    failures.extend(_paired_ordering_failures(receipt))
    failures.extend(_observed_scoring_failures(receipt))
    failures.extend(_pin_failures(receipt))
    return failures
