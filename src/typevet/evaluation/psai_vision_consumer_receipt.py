"""Receipt serialization and acceptance for the PSAI consumer harness ([#177][i177]).

Examples:
    ```python
    from typevet.evaluation.psai_vision_consumer_receipt import (
        evaluate_consumer_receipt_acceptance,
        serialize_answer,
    )
    from typevet.domain.judgment_answers import NoulAnswer

    payload = serialize_answer(NoulAnswer(noul=0.5))
    assert payload["kind"] == "Noul"
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_harness][]: orchestration

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from typevet.domain.judgment_answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from typevet.evaluation.psai_vision_consumer_outcomes import expected_outcome_failures


def serialize_answer(answer: Answer) -> dict[str, Any]:
    """Map a public ``Answer`` to a JSON-serializable dict.

    Args:
        answer: Typed judgment answer.

    Returns:
        Dict with ``kind`` and type-specific fields.

    Raises:
        TypeError: When ``answer`` is not a known ``Answer`` subtype.
    """
    if isinstance(answer, NoulAnswer):
        return {"kind": "Noul", "noul": answer.noul}
    if isinstance(answer, ChoiceAnswer):
        return {
            "kind": "Choice",
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": dict(answer.probabilities),
        }
    if isinstance(answer, ScoreAnswer):
        return {
            "kind": "Score",
            "score": answer.score,
            "confidence": answer.confidence,
            "legend": {str(k): v for k, v in answer.legend.items()},
            "probabilities": {str(k): v for k, v in answer.probabilities.items()},
        }
    msg = f"unsupported answer type: {type(answer)!r}"
    raise TypeError(msg)


def consumer_receipt_basename(
    *,
    protocol_revision: int,
    wheel_sha256: str | None,
    model_id: str,
) -> str:
    """Return a stable receipt filename for one proof configuration.

    Args:
        protocol_revision: Frozen protocol revision integer.
        wheel_sha256: Optional wheel digest from the environment.
        model_id: Model id under test.

    Returns:
        Filename including ``.json`` suffix.
    """
    wheel_part = (wheel_sha256 or "unknown-wheel")[:16]
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)[:48]
    return f"consumer-receipt-p{protocol_revision}-{wheel_part}-{model_slug}.json"


def resolve_receipt_write_path(
    out_dir: Path,
    *,
    protocol_revision: int,
    wheel_sha256: str | None,
    model_id: str,
) -> Path:
    """Pick a non-colliding receipt path under ``out_dir``.

    Args:
        out_dir: Directory for receipts.
        protocol_revision: Frozen protocol revision.
        wheel_sha256: Optional wheel digest.
        model_id: Model id under test.

    Returns:
        Path that does not yet exist (appends ``-attempt-N`` when needed).

    Raises:
        OSError: When no free attempt suffix is available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / consumer_receipt_basename(
        protocol_revision=protocol_revision,
        wheel_sha256=wheel_sha256,
        model_id=model_id,
    )
    if not base.exists():
        return base
    for attempt in range(1, 1000):
        candidate = base.with_name(base.stem + f"-attempt-{attempt}" + base.suffix)
        if not candidate.exists():
            return candidate
    msg = f"could not allocate receipt path under {out_dir}"
    raise OSError(msg)


def evaluate_consumer_receipt_acceptance(
    receipt: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Check frozen acceptance conditions on a saved receipt.

    Args:
        receipt: Consumer proof receipt dict.

    Returns:
        ``(accepted, failure_messages)``.
    """
    failures: list[str] = []
    capability = receipt.get("capability")
    if isinstance(capability, Mapping) and capability.get("vision") is False:
        failures.append("capability.vision is false")
    negative = receipt.get("unsupported_capability_negative")
    if isinstance(negative, Mapping) and not negative.get("ok"):
        failures.append("unsupported_capability_negative did not pass")
    for pair in receipt.get("paired_ordering") or []:
        if isinstance(pair, Mapping) and not pair.get("ordered"):
            uid = pair.get("unique_data_id", "?")
            failures.append(f"paired_ordering failed for row {uid!r}")
    matrix = receipt.get("matrix_rows") or []
    for row in matrix:
        if not isinstance(row, Mapping):
            continue
        answers = row.get("answers")
        if not isinstance(answers, Mapping) or not answers:
            failures.append("matrix row missing answers map")
            break
    expected_j = int(receipt.get("judgment_call_count", 0))
    if expected_j and len(matrix) != expected_j:
        failures.append(
            f"matrix_rows {len(matrix)} != judgment_call_count {expected_j}"
        )
    failures.extend(expected_outcome_failures(receipt))
    return (not failures, failures)


def write_receipt_json(path: Path, receipt: Mapping[str, Any]) -> None:
    """Write ``receipt`` as indented JSON to ``path``.

    Args:
        path: Destination file.
        receipt: Serializable receipt mapping.
    """
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
