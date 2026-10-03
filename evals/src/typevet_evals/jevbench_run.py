"""Run pinned JevBench tasks through the configured typevet judgment session.

Upstream owns scoring and durable raw/results output. The manifest identifies
this harness by commit because upstream package version strings disagree.

Attributes:
    UPSTREAM_COMMIT (str): Locked JevBench source revision.

Examples:
    ```console
    TYPEVET_BACKEND=fake uv run python -m typevet_evals.jevbench_run --help
    ```

See Also:
    - [typevet_evals.jevbench][]: Typed provider adapter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.tasks import load_jsonl, sha256_file

from typevet.adapters.inbound import load_backend, open_judgment
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort
from typevet_evals.jevbench import SystemOneAdapter

UPSTREAM_COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
__all__ = ["UPSTREAM_COMMIT", "main"]


def _parser() -> argparse.ArgumentParser:
    """Define the public module options.

    Returns:
        Parser for task input, evidence paths and upstream budget controls.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tasks", "results", "raw-dir", "ledger", "manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cap-usd", type=float, default=15.0)
    parser.add_argument("--reserve-usd", type=float, default=0.02)
    return parser


def _execute(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    """Compose the owned session and borrowed adapter with upstream.

    Args:
        args: Parsed public options.
        manifest: Evidence updated before and after upstream execution.

    Returns:
        Zero for a complete valid run, one for failed or unattempted tasks.
    """
    tasks = load_jsonl(str(args.tasks))
    manifest["dataset_sha256"] = sha256_file(str(args.tasks))
    manifest["dataset_count"] = len(tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    manifest["requested_count"] = len(tasks)
    manifest["settings"]["backend"] = load_backend()
    with open_judgment() as session:
        manifest["session_model"] = session.model
        adapter = SystemOneAdapter(TypevetSystemOnePort(session.port), model=args.model)
        runner = Runner(
            adapter,
            Ledger(args.ledger, cap_usd=args.cap_usd),
            args.raw_dir,
            default_reserve_usd=args.reserve_usd,
        )
        records = runner.run_all(tasks, results_path=args.results)
    manifest["attempted_count"] = len(records)
    manifest["resolved_models"] = sorted({record["model"] for record in records})
    complete = bool(tasks) and len(records) == len(tasks)
    return (
        0
        if complete and all(record["ok"] and record["valid"] for record in records)
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    """Run tasks and always retain a manifest for an accepted output path.

    Args:
        argv: Explicit arguments, or None for the process command line.

    Returns:
        Zero on completion, one on run failure, or two for unsafe paths.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.cap_usd, args.reserve_usd)
    ):
        parser.error("budget controls must be finite and nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.results.exists() or args.raw_dir.exists() or args.manifest.exists():
        parser.error("results, raw-dir and manifest must be fresh paths")
    paths = [args.tasks, args.results, args.raw_dir, args.ledger, args.manifest]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("input and evidence paths must be distinct")
    manifest: dict[str, Any] = {
        "upstream_commit": UPSTREAM_COMMIT,
        "requested_model": args.model,
        "resolved_models": [],
        "requested_count": 0,
        "attempted_count": 0,
        "settings": {
            "limit": args.limit,
            "cap_usd": args.cap_usd,
            "reserve_usd": args.reserve_usd,
        },
        "cost_basis": "unknown",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as stream:
        try:
            code = _execute(args, manifest)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            manifest["error"] = type(exc).__name__
            code = 1
        manifest["exit_code"] = code
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
