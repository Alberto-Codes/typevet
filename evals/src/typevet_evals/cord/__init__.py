"""CORD expense evaluation family ([#161][i161], moved in #256 E6).

The family runs the CORD expense-claim smoke and its offline acceptance. It
gates the served capability and the image attachment, judges each arm, counts
judgment calls and checks a combined receipt against the #161 semantic
floors. This package re-exports the names that callers outside the package
use.

Attributes:
    __all__ (list[str]): Public CORD names re-exported from the submodules.

Examples:
    ```python
    import json
    from pathlib import Path

    from typevet_evals.cord import accept_combined_receipt

    receipt = json.loads(
        Path(
            "tests/fixtures/cord/semantic_acceptance/labeled_synthetic_pass.json"
        ).read_text(encoding="utf-8")
    )
    result = accept_combined_receipt(receipt)
    ```

See Also:
    - [typevet_evals.cord.semantic_acceptance][]: #161 acceptance floors
    - [typevet_evals.cord.semantic_acceptance_report][]: operator report
    - [typevet_evals.cord.semantic_metrics][]: shared confusion metrics
    - [typevet_evals.cord.expense_smoke][]: capability gate and attachment
    - [typevet_evals.cord.expense_live_harness][]: live orchestration entry
    - [typevet_evals.cord.expense_receipt_requirement][]: per-arm judging
    - [typevet_evals.cord.expense_call_accounting][]: judgment call totals
    - [typevet_evals.cli.cord_semantic_acceptance][]: operator CLI

[i161]: https://github.com/Alberto-Codes/typevet/issues/161
"""

from __future__ import annotations

from typevet_evals.cord.expense_call_accounting import (
    cord_expense_smoke_request_totals,
    summarize_cord_expense_judgment_calls,
)
from typevet_evals.cord.expense_live_harness import (
    orchestrate_cord_expense_live_smoke,
)
from typevet_evals.cord.expense_receipt_requirement import judge_cord_expense_arm
from typevet_evals.cord.expense_smoke import (
    assert_cord_expense_attachment,
    assert_cord_expense_live_smoke_gate,
    resolve_cord_expense_attachment_profile,
)
from typevet_evals.cord.semantic_acceptance import accept_combined_receipt
from typevet_evals.cord.semantic_acceptance_report import (
    evaluate_combined_receipt,
    format_semantic_acceptance_report,
)

__all__ = [
    "accept_combined_receipt",
    "assert_cord_expense_attachment",
    "assert_cord_expense_live_smoke_gate",
    "cord_expense_smoke_request_totals",
    "evaluate_combined_receipt",
    "format_semantic_acceptance_report",
    "judge_cord_expense_arm",
    "orchestrate_cord_expense_live_smoke",
    "resolve_cord_expense_attachment_profile",
    "summarize_cord_expense_judgment_calls",
]
