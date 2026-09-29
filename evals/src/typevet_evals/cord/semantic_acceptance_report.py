"""Human-readable reporting for CORD semantic acceptance ([#184][i184]).

Examples:
    ```python
    from typevet_evals.cord.semantic_acceptance import accept_combined_receipt
    from typevet_evals.cord.semantic_acceptance_report import (
        format_semantic_acceptance_report,
    )

    outcome = accept_combined_receipt({"cases": [], "combined": {}})
    ```

See Also:
    - [typevet_evals.cord.semantic_acceptance][]: acceptance floors
    - [typevet_evals.cli.cord_semantic_acceptance][]: operator CLI

[i184]: https://github.com/Alberto-Codes/typevet/issues/184
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typevet_evals.cord.semantic_acceptance import (
    CheckStatus,
    SemanticAcceptanceOutcome,
    accept_combined_receipt,
)


def evaluate_combined_receipt(receipt: Mapping[str, Any]) -> SemanticAcceptanceOutcome:
    """Run frozen #161 revision 1 floors on one parsed receipt mapping.

    Args:
        receipt: Saved combined outcome with ``cases`` and ``combined`` arms.

    Returns:
        Acceptance outcome with per-floor checks.

    Raises:
        ValueError: When coverage, labels or gold verdicts are invalid.
    """
    return accept_combined_receipt(receipt)


def _status_label(status: CheckStatus) -> str:
    if status is CheckStatus.PASS:
        return "PASS"
    if status is CheckStatus.FAIL:
        return "FAIL"
    return "NOT_COMPUTABLE"


def _format_measured(measured: float | None) -> str:
    if measured is None:
        return "—"
    return f"{measured:.4g}"


def format_semantic_acceptance_report(outcome: SemanticAcceptanceOutcome) -> str:
    """Render a PASS/FAIL table and summary line for operators.

    Args:
        outcome: Result from ``accept_combined_receipt`` or
            ``evaluate_combined_receipt``.

    Returns:
        Multi-line text suitable for stdout.
    """
    lines = [
        "| check | bound | limit | measured | n | status |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines.extend(
        "| "
        f"{check.name} | {check.bound.value} | {check.limit:.2g} | "
        f"{_format_measured(check.measured)} | {check.denominator} | "
        f"{_status_label(check.status)} |"
        for check in outcome.checks
    )
    verdict = "true" if outcome.accepted else "false"
    lines.append("")
    lines.append(f"claims: {outcome.claims}")
    lines.append(f"accepted: {verdict}")
    if outcome.failures:
        lines.append("")
        lines.append("failures:")
        lines.extend(f"- {reason}" for reason in outcome.failures)
    return "\n".join(lines)
