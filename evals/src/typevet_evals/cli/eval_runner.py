r"""CLI entry for opt-in loader eval against a live llama.cpp router (#98).

Examples:
    ```console
    $ uv run python -m typevet_evals.cli.eval_runner --dataset boolq --limit 2
    dataset=boolq\tmetric=exact_match\tlimit=2\tattempted=2\tschema_valid=2\tgold_match=1
    ```

    Proof runs that must not silently skip:

    ```console
    $ uv run python -m typevet_evals.cli.eval_runner --require-live --dataset boolq --limit 2
    ```

See Also:
    - [typevet_evals.runner.core][]: core runner
    - [typevet_evals.runner.live_gate][]: skip when router unset
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from typevet.adapters.inbound.settings import llama_cpp_adapter, load_llama_settings
from typevet.ports.generation import GenerationPort
from typevet_evals.runner.core import run_eval_tasks
from typevet_evals.runner.datasets import (
    SUPPORTED_DATASETS,
    EvalDatasetName,
    load_eval_tasks,
)
from typevet_evals.runner.live_gate import live_skip_reason
from typevet_evals.runner.report import EvalRunReport, format_report

_EXIT_USAGE = 2
_EXIT_REQUIRE_LIVE = 1


def _as_dataset(name: str) -> EvalDatasetName:
    if name in SUPPORTED_DATASETS:
        return name
    msg = f"unsupported eval dataset {name!r}"
    raise ValueError(msg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny Banking77 or BoolQ slice through GenerationPort. "
            "Counts structure-valid outputs and gold agreement only (not ECE)."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        action="append",
        dest="datasets",
        help="Loader to run (repeat for both banking77 and boolq)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Row cap per dataset (balanced sampling when supported)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic balance seed for loaders",
    )
    parser.add_argument(
        "--require-live",
        action="store_true",
        help=(
            "Nonzero exit when config, model, workload, or schema-valid "
            "completion is missing (default: skip successfully)"
        ),
    )
    return parser


def _limit_error(limit: int) -> str | None:
    if limit <= 0:
        return f"--limit must be positive, got {limit}"
    return None


def _require_live_failure_reason(
    *,
    skip: str | None,
    model: str | None,
    reports: list[EvalRunReport],
) -> str | None:
    if skip is not None:
        return skip
    if model is None:
        return "TYPEVET_LLAMA__DEFAULT_MODEL (or TYPEVET_GEMMA_MODEL) not set"
    for report in reports:
        if report.attempted == 0:
            return f"empty workload for dataset={report.dataset!r}"
        if report.schema_valid != report.attempted:
            return (
                f"schema-valid calls incomplete for dataset={report.dataset!r} "
                f"({report.schema_valid}/{report.attempted})"
            )
    return None


def _collect_reports(
    *,
    port: GenerationPort,
    datasets: list[str],
    limit: int,
    seed: int,
    model: str,
) -> list[EvalRunReport]:
    reports: list[EvalRunReport] = []
    for dataset in datasets:
        tasks = load_eval_tasks(_as_dataset(dataset), limit=limit, seed=seed)
        if not tasks:
            reports.append(
                EvalRunReport(
                    dataset=dataset,
                    metric_name="exact_match",
                    limit=limit,
                    attempted=0,
                    schema_valid=0,
                    gold_match=0,
                )
            )
            continue
        reports.append(run_eval_tasks(port, tasks, model=model))
    return reports


def main(argv: list[str] | None = None) -> int:
    """Run live eval slices and print tab-separated reports.

    Args:
        argv: CLI args; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success or intentional skip, ``1`` when ``--require-live`` fails,
        ``2`` on usage errors.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    limit_err = _limit_error(args.limit)
    if limit_err is not None:
        print(f"error: {limit_err}", file=sys.stderr)
        return _EXIT_USAGE
    datasets = args.datasets or ["boolq"]
    settings = load_llama_settings()
    skip = live_skip_reason(settings)
    model = settings.default_model
    if args.require_live:
        reason = _require_live_failure_reason(skip=skip, model=model, reports=[])
        if reason is not None:
            print(f"require-live: {reason}", file=sys.stderr)
            return _EXIT_REQUIRE_LIVE
    elif skip is not None:
        print(f"skip: {skip}", file=sys.stderr)
        return 0
    if model is None:
        print(
            "skip: TYPEVET_LLAMA__DEFAULT_MODEL (or TYPEVET_GEMMA_MODEL) not set",
            file=sys.stderr,
        )
        return 0
    live_settings = replace(settings, timeout=max(settings.timeout, 600.0))
    with llama_cpp_adapter(live_settings) as port:
        reports = _collect_reports(
            port=port,
            datasets=datasets,
            limit=args.limit,
            seed=args.seed,
            model=model,
        )
    if args.require_live:
        reason = _require_live_failure_reason(skip=None, model=model, reports=reports)
        if reason is not None:
            print(f"require-live: {reason}", file=sys.stderr)
            return _EXIT_REQUIRE_LIVE
    for report in reports:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
