"""Live consumer receipt payload assembly ([#177][i177]).

Examples:
    ```python
    from typevet.evaluation.psai_vision_consumer_live_receipt import (
        build_live_receipt_payload,
    )

    assert callable(build_live_receipt_payload)
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_live][]: orchestration

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typevet.evaluation.datasets.psai_vision_controls import PairedOrdering
from typevet.evaluation.psai_vision_consumer_accounting import (
    ANNOTATION_QUESTIONS_PER_JUDGE_CALL,
    ConsumerCallCounts,
)
from typevet.evaluation.psai_vision_consumer_harness import PROTOCOL_REVISION
from typevet.evaluation.psai_vision_consumer_live_router import ConsumerLiveMatrixResult
from typevet.evaluation.psai_vision_consumer_offline import (
    consumer_fixture_identity_pins,
)


@dataclass(frozen=True, slots=True)
class LiveReceiptContext:
    """Inputs for ``build_live_receipt_payload``.

    Attributes:
        fixture_root (Path): Committed fixture directory.
        plan (ConsumerCallCounts): Frozen scheduled call counts.
        matrix (ConsumerLiveMatrixResult): Live matrix outputs.
        model (str): Model id under test.
        wheel_sha256 (str | None): Optional wheel digest.
        typevet_install_path (str): Resolved ``typevet.__file__`` path.
        evidence_kind (str): Evidence label.
        negative (dict[str, Any]): Unsupported-template probe outcome.
        pairs (Sequence[PairedOrdering]): Paired ordering summary rows.
        git_head (str): Repository ``HEAD`` at assembly time.

    Examples:
        ```python
        from typevet.evaluation.psai_vision_consumer_live_receipt import (
            LiveReceiptContext,
        )

        assert LiveReceiptContext.__dataclass_fields__
        ```
    """

    fixture_root: Path
    plan: ConsumerCallCounts
    matrix: ConsumerLiveMatrixResult
    model: str
    wheel_sha256: str | None
    typevet_install_path: str
    evidence_kind: str
    negative: dict[str, Any]
    pairs: Sequence[PairedOrdering]
    git_head: str


def build_live_receipt_payload(context: LiveReceiptContext) -> dict[str, Any]:
    """Build the JSON receipt body for one successful live consumer proof.

    Args:
        context: Live receipt assembly inputs.

    Returns:
        Serializable receipt mapping (not yet acceptance-checked).
    """
    fixture_root = context.fixture_root
    plan = context.plan
    matrix = context.matrix
    ledger = matrix.ledger
    try:
        package_version = importlib.metadata.version("typevet")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    payload: dict[str, Any] = {
        "consumer_live_protocol_revision": PROTOCOL_REVISION,
        "evidence_kind": context.evidence_kind,
        "git_head": context.git_head,
        "wheel_sha256": context.wheel_sha256,
        "typevet_version": package_version,
        "typevet_install_path": context.typevet_install_path,
        "model": context.model,
        "require_live": True,
        "questions_per_judge_call_annotation": ANNOTATION_QUESTIONS_PER_JUDGE_CALL,
        "judgment_call_count": plan.judgment_calls,
        "scoring_request_count": plan.scoring_requests,
        "scoring_requests_observed": ledger.scoring_requests,
        "auxiliary_http_count": ledger.auxiliary_http_total,
        "auxiliary_http_metadata_count": ledger.auxiliary_metadata_http,
        "auxiliary_http_tokenizer_count": ledger.auxiliary_tokenizer_http,
        "failed_attempts": ledger.failed_attempts,
        "matrix_rows": matrix.matrix_rows,
        "health": matrix.health,
        "capability": {
            "vision": matrix.capability.vision,
            "marker": matrix.capability.marker,
        },
        "served_template": matrix.served.name,
        "unsupported_capability_negative": context.negative,
        "paired_ordering": [
            {
                "unique_data_id": p.unique_data_id,
                "ordered": p.ordered,
                "margin": p.margin,
            }
            for p in context.pairs
        ],
        "elapsed_s": matrix.elapsed_s,
        "fixture_root": str(fixture_root.resolve()),
    }
    payload.update(matrix.identity)
    payload.update(consumer_fixture_identity_pins(fixture_root))
    return payload
