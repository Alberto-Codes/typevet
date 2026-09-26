"""Aggregated counts for a typed-loader eval slice (#98).

Measures structure-valid outputs and gold agreement only. This is not
calibration or ECE ([#50](https://github.com/Alberto-Codes/typevet/issues/50)).

Examples:
    ```python
    from typevet.evaluation.runner.report import EvalRunReport, merge_reports

    left = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=2,
        attempted=2,
        schema_valid=2,
        gold_match=1,
    )
    merged = merge_reports([left])
    assert merged.attempted == 2
    ```

See Also:
    - [typevet.evaluation.runner.core][]: GenerationPort driver
    - [typevet.evaluation.runner.datasets][]: Banking77 and BoolQ task specs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

DEFAULT_METRIC_EXACT_MATCH: Final[str] = "exact_match"
DEFAULT_METRIC_NOUL_AGREEMENT: Final[str] = "noul_agreement"


@dataclass(frozen=True, slots=True)
class EvalRunReport:
    """Summary for one dataset slice driven through ``GenerationPort``.

    Attributes:
        dataset (str): Loader id (``banking77`` or ``boolq``).
        metric_name (str): Gold comparison name (``exact_match`` or
            ``noul_agreement``).
        limit (int): Requested row cap for the slice.
        attempted (int): Generation calls made.
        schema_valid (int): Calls that returned a schema-valid result.
        gold_match (int): Schema-valid outputs that matched gold.

    Examples:
        ```python
        from typevet.evaluation.runner.report import EvalRunReport

        EvalRunReport(
            dataset="banking77",
            metric_name="noul_agreement",
            limit=4,
            attempted=4,
            schema_valid=3,
            gold_match=2,
        )
        ```
    """

    dataset: str
    metric_name: str
    limit: int
    attempted: int
    schema_valid: int
    gold_match: int


def merge_reports(reports: list[EvalRunReport]) -> EvalRunReport:
    """Sum counts across reports that share dataset and metric.

    Args:
        reports: Non-empty list with identical ``dataset`` and ``metric_name``.

    Returns:
        Combined report; ``limit`` is the sum of slice limits.

    Raises:
        ValueError: When ``reports`` is empty or keys disagree.
    """
    if not reports:
        msg = "reports must be non-empty"
        raise ValueError(msg)
    dataset = reports[0].dataset
    metric_name = reports[0].metric_name
    for report in reports[1:]:
        if report.dataset != dataset or report.metric_name != metric_name:
            msg = "merge_reports requires the same dataset and metric_name"
            raise ValueError(msg)
    return EvalRunReport(
        dataset=dataset,
        metric_name=metric_name,
        limit=sum(r.limit for r in reports),
        attempted=sum(r.attempted for r in reports),
        schema_valid=sum(r.schema_valid for r in reports),
        gold_match=sum(r.gold_match for r in reports),
    )


def format_report(report: EvalRunReport) -> str:
    """Return a single-line human summary for logs and CLI output.

    Args:
        report: Completed slice summary.

    Returns:
        Tab-separated counters with dataset and metric labels.
    """
    return (
        f"dataset={report.dataset}\tmetric={report.metric_name}\tlimit={report.limit}\t"
        f"attempted={report.attempted}\tschema_valid={report.schema_valid}\t"
        f"gold_match={report.gold_match}"
    )
