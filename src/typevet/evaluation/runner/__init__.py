"""Loader eval runner: dataset specs, live gate and run reports (#147).

Examples:
    ```python
    from typevet.evaluation.runner import load_eval_tasks, run_eval_tasks
    ```

See Also:
    - [typevet.evaluation.runner.core][]: Run tasks through a generation port
    - [typevet.evaluation.runner.datasets][]: Supported datasets and task specs
    - [typevet.evaluation.runner.live_gate][]: Skip reasons for live runs
    - [typevet.evaluation.runner.report][]: Report shape and formatting
    - [typevet.adapters.inbound.eval_cli][]: CLI entry for these runs

Attributes:
    SUPPORTED_DATASETS (tuple): Dataset names the runner can load.
    DEFAULT_METRIC_EXACT_MATCH (str): Metric name for exact-match scoring.
    DEFAULT_METRIC_NOUL_AGREEMENT (str): Metric name for noul agreement.
    EvalDatasetName (type): Literal alias for supported dataset names.
    EvalTaskSpec (type): One prompt, schema and gold value to run.
    EvalRunReport (type): Counts for one dataset run.
    load_eval_tasks (function): Build task specs for a dataset slice.
    run_eval_tasks (function): Run task specs through a generation port.
    live_skip_reason (function): Skip message when the router is unavailable.
    format_report (function): Render one report as a tab-separated line.
    merge_reports (function): Add report counts for the same dataset.
"""

from typevet.evaluation.runner.core import run_eval_tasks
from typevet.evaluation.runner.datasets import (
    SUPPORTED_DATASETS,
    EvalDatasetName,
    EvalTaskSpec,
    load_eval_tasks,
)
from typevet.evaluation.runner.live_gate import live_skip_reason
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
    "SUPPORTED_DATASETS",
    "EvalDatasetName",
    "EvalRunReport",
    "EvalTaskSpec",
    "format_report",
    "live_skip_reason",
    "load_eval_tasks",
    "merge_reports",
    "run_eval_tasks",
]
