r"""CLI entry for opt-in loader eval against a live llama.cpp router (#98).

Examples:
    ```console
    $ uv run python -m typevet.eval_runner_cli --dataset boolq --limit 2
    dataset=boolq\tmetric=exact_match\tlimit=2\tattempted=2\tschema_valid=2\tgold_match=1
    ```

See Also:
    - [typevet.eval_runner][]: core runner
    - [typevet.eval_runner_live_gate][]: skip when router unset
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from typevet.adapters.inbound.settings import llama_cpp_adapter, load_llama_settings
from typevet.eval_runner import run_eval_tasks
from typevet.eval_runner_datasets import (
    SUPPORTED_DATASETS,
    EvalDatasetName,
    load_eval_tasks,
)
from typevet.eval_runner_live_gate import live_skip_reason
from typevet.eval_runner_report import format_report


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run live eval slices and print tab-separated reports.

    Args:
        argv: CLI args; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success or intentional skip, ``2`` on usage errors.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    datasets = args.datasets or ["boolq"]
    settings = load_llama_settings()
    skip = live_skip_reason(settings)
    if skip is not None:
        print(f"skip: {skip}", file=sys.stderr)
        return 0
    model = settings.default_model
    if model is None:
        print(
            "skip: TYPEVET_LLAMA__DEFAULT_MODEL (or TYPEVET_GEMMA_MODEL) not set",
            file=sys.stderr,
        )
        return 0
    live_settings = replace(settings, timeout=max(settings.timeout, 600.0))
    reports = []
    with llama_cpp_adapter(live_settings) as port:
        for dataset in datasets:
            tasks = load_eval_tasks(
                _as_dataset(dataset), limit=args.limit, seed=args.seed
            )
            reports.append(run_eval_tasks(port, tasks, model=model))
    for report in reports:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
