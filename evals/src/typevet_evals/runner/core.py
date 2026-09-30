"""Opt-in eval runner over typed loaders via ``GenerationPort`` (#98).

Reports attempted, schema-valid, and gold-match counts. Does not compute
calibration or ECE ([#50](https://github.com/Alberto-Codes/typevet/issues/50)).
A task whose schema fails the request schema check counts as a failed task;
the run continues (#238).

Examples:
    Offline wiring with a fake port:

    ```python
    from pathlib import Path

    from typevet.adapters.outbound.fake import FakeGenerationAdapter
    from typevet_evals.runner.core import run_eval_tasks
    from typevet_evals.runner.datasets import load_eval_tasks

    jsonl = Path("tests/fixtures/boolq/validation_smoke.jsonl").read_text()
    tasks = load_eval_tasks("boolq", limit=2, boolq_jsonl_text=jsonl)
    port = FakeGenerationAdapter(responder=lambda req: {"answer": "yes"})
    report = run_eval_tasks(port, tasks, model="fake")
    assert report.attempted == 1
    ```

See Also:
    - [typevet_evals.runner.datasets][]: Task loading
    - [typevet_evals.runner.report][]: Result aggregation
    - [typevet.ports.generation][]: GenerationPort protocol
"""

from __future__ import annotations

from collections.abc import Sequence

from typevet.domain.errors import GenerationError
from typevet.domain.models import GenerationRequest
from typevet.ports.generation import GenerationPort
from typevet_evals.runner.datasets import EvalTaskSpec
from typevet_evals.runner.report import (
    DEFAULT_METRIC_EXACT_MATCH,
    DEFAULT_METRIC_NOUL_AGREEMENT,
    EvalRunReport,
)

# Message prefix of the schema check in ``check_request_schema`` (#238).
_SCHEMA_CHECK_PREFIX = "schema is not a valid JSON Schema:"


def _metric_for_dataset(dataset: str) -> str:
    if dataset == "banking77":
        return DEFAULT_METRIC_NOUL_AGREEMENT
    return DEFAULT_METRIC_EXACT_MATCH


def run_eval_tasks(
    port: GenerationPort,
    tasks: Sequence[EvalTaskSpec],
    *,
    model: str,
) -> EvalRunReport:
    """Drive ``tasks`` through ``port.generate`` and aggregate counts.

    A ``GenerationError`` or a schema-check ``ValueError`` (message starts
    with ``schema is not a valid JSON Schema:``) counts as a failed task,
    and the run continues.

    Args:
        port: ``GenerationPort`` implementation (fake or llama.cpp).
        tasks: Slice from :func:`typevet_evals.runner.datasets.load_eval_tasks`.
        model: Model id passed on each ``GenerationRequest``.

    Returns:
        Summary with attempted, schema-valid, and gold-match totals.

    Raises:
        ValueError: When ``tasks`` is empty or mixes datasets, or when the
            port raises a ``ValueError`` that is not the schema check.
    """
    if not tasks:
        msg = "tasks must be non-empty"
        raise ValueError(msg)
    dataset = tasks[0].dataset
    for task in tasks[1:]:
        if task.dataset != dataset:
            msg = "run_eval_tasks requires a single dataset per call"
            raise ValueError(msg)

    attempted = 0
    schema_valid = 0
    gold_match = 0
    for task in tasks:
        attempted += 1
        request = GenerationRequest(
            prompt=task.prompt,
            schema=task.schema,
            model=model,
        )
        try:
            result = port.generate(request)
        except GenerationError:
            continue
        except ValueError as exc:
            if not str(exc).startswith(_SCHEMA_CHECK_PREFIX):
                raise
            continue
        schema_valid += 1
        predicted = result.value.get(task.noul_field)
        if predicted == task.gold:
            gold_match += 1

    return EvalRunReport(
        dataset=dataset,
        metric_name=_metric_for_dataset(dataset),
        limit=len(tasks),
        attempted=attempted,
        schema_valid=schema_valid,
        gold_match=gold_match,
    )
