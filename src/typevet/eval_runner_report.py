"""Compatibility shim for eval run reports (#147).

Prefer importing from ``typevet.evaluation.runner.report`` in new code.

Examples:
    ```python
    from typevet.eval_runner_report import format_report
    ```

See Also:
    - [typevet.evaluation.runner.report][]: New home for this module
"""

from typevet.evaluation.runner.report import (
    DEFAULT_METRIC_EXACT_MATCH,
    DEFAULT_METRIC_NOUL_AGREEMENT,
    EvalRunReport,
    format_report,
    merge_reports,
)

__all__ = [
    "DEFAULT_METRIC_EXACT_MATCH",
    "DEFAULT_METRIC_NOUL_AGREEMENT",
    "EvalRunReport",
    "format_report",
    "merge_reports",
]
