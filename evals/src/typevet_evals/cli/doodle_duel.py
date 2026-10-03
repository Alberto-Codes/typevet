r"""Judge Quick, Draw! doodles with one ``Choice`` each and write a receipt (#412).

The command takes the first ``--per-category`` recognized doodles of each
category from the Quick, Draw! cache, streaming a category file on a cache
miss only. It renders each doodle to a 512 px PNG, judges one ``Choice``
over the categories per doodle in a seeded order, writes a key-free receipt
and prints the accuracy. The receipt holds key ids, labels, every option
probability, metrics and pins, never image bytes. The command refuses an
existing receipt path before any request.

``TYPEVET_BACKEND`` selects the backend, as in ``open_judgment``
(``llama_cpp`` by default, ``vllm`` or ``fake``). ``--require-live`` refuses
``fake``. A vLLM run needs ``TYPEVET_VLLM_MODEL_REVISION``.
``TYPEVET_IMAGE_CONCURRENCY`` sets the judgments in flight.
``TYPEVET_GIT_STATUS_PORCELAIN`` carries the porcelain status text for the
working-tree fingerprint. Exit codes: ``0`` when every doodle ran, ``1`` on a
refusal (a failed download on a cache miss included) or a stopped run, ``2``
on a usage error (a repeated ``--categories`` name included).

Examples:
    ```console
    $ uv run python -m typevet_evals.cli.doodle_duel \\
        --receipt out.json --per-category 5
    accuracy 0.5
    ```

See Also:
    - [typevet_evals.doodle_duel][]: request, render, run and receipt
    - [typevet_evals.datasets.quickdraw][]: first-N loader and cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from typevet.adapters.inbound.backend_settings import (
    load_backend,
    load_vllm_settings,
    open_judgment,
)
from typevet_evals.datasets.quickdraw import STROKES_URL, Doodle, fetch_doodles
from typevet_evals.doodle_duel import (
    DOODLE_CATEGORIES,
    DOODLE_QUESTION,
    DoodleRequest,
    DoodleRun,
    build_doodle_receipt,
    build_doodle_request,
    doodle_questions,
    render_strokes,
    run_doodle_duel,
    shuffle_rows,
)
from typevet_evals.experiment_identity import (
    PromptSpec,
    RunIdentityStart,
    RuntimeBuild,
    WorkingTreeState,
    begin_run_identity,
    capture_working_tree_at_run_start,
    finalize_experiment_identity,
    read_baseline_commit,
    snapshot_evaluated_inputs,
    write_receipt_exclusive,
)
from typevet_evals.face_match import (
    ensure_key_free,
    image_concurrency,
    served_weights_pins,
)
from typevet_evals.runner.live_gate import live_backend

_EXIT_OK = 0
_EXIT_REFUSED = 1
_EXIT_USAGE = 2
_MIN_CATEGORIES = 2
_EVALS_SRC = Path(__file__).resolve().parents[1]
_CODE_PATHS = {
    "quickdraw": _EVALS_SRC / "datasets" / "quickdraw.py",
    **{
        name: _EVALS_SRC / "doodle_duel" / f"{name}.py"
        for name in ("categories", "render", "request", "runner")
    },
}


def _categories(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DOODLE_CATEGORIES
    names = [name.strip() for name in raw.split(",") if name.strip()]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        msg = f"--categories repeats: {repeated}"
        raise argparse.ArgumentTypeError(msg)
    asked = set(names)
    unknown = sorted(asked - set(DOODLE_CATEGORIES))
    if unknown or len(asked) < _MIN_CATEGORIES:
        msg = f"--categories needs 2 or more of DOODLE_CATEGORIES; unknown: {unknown}"
        raise argparse.ArgumentTypeError(msg)
    return tuple(name for name in DOODLE_CATEGORIES if name in asked)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge Quick, Draw! doodles with one Choice each.",
    )
    parser.add_argument("--receipt", type=Path, required=True, help="New file")
    parser.add_argument("--per-category", type=int, default=5, help="Default 5")
    parser.add_argument("--seed", type=int, default=0, help="Run order seed")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache dir")
    parser.add_argument(
        "--categories", type=_categories, default=DOODLE_CATEGORIES, help="a,b,..."
    )
    parser.add_argument(
        "--require-live", action="store_true", help="Refuse TYPEVET_BACKEND=fake"
    )
    return parser


def _requests(
    doodles: Sequence[Doodle], categories: Sequence[str], seed: int
) -> list[DoodleRequest]:
    requests = [
        build_doodle_request(d, png=render_strokes(d.strokes), categories=categories)
        for d in doodles
    ]
    return shuffle_rows(requests, seed, key=lambda request: request.key_id)


def _working_tree(environ: Mapping[str, str]) -> WorkingTreeState:
    root = Path.cwd()
    porcelain = environ.get("TYPEVET_GIT_STATUS_PORCELAIN")
    if porcelain is None:
        return WorkingTreeState(
            read_baseline_commit(root),
            True,
            ("dirty-unknown-without-porcelain",),
            "unknown",
        )
    return capture_working_tree_at_run_start(root, porcelain=porcelain)


def _prompt_spec(categories: Sequence[str]) -> PromptSpec:
    question = doodle_questions(categories)[DOODLE_QUESTION]
    criteria = {str(k): str(v) for k, v in question.criteria.items()}
    return PromptSpec(
        DOODLE_QUESTION, tuple(criteria), str(question.instructions), criteria
    )


def _judge(
    environ: Mapping[str, str],
    requests: Sequence[DoodleRequest],
    tree: WorkingTreeState,
) -> tuple[DoodleRun, str, RunIdentityStart]:
    with open_judgment(environ) as session:
        served = getattr(session, "served", None)
        start = begin_run_identity(
            repo_root=Path.cwd(),
            runtime=RuntimeBuild(
                session.model, served.value if served else "unknown", "unknown"
            ),
            working_tree=tree,
        )
        run = run_doodle_duel(
            session.port,
            requests,
            session.model,
            concurrency=image_concurrency(environ),
        )
        return run, session.model, start


def _pins(
    args: argparse.Namespace,
    doodles: Sequence[Doodle],
    environ: Mapping[str, str],
    backend: str,
) -> dict[str, object]:
    key_ids: dict[str, list[str]] = {}
    for doodle in doodles:
        key_ids.setdefault(doodle.word, []).append(doodle.key_id)
    return {
        "finished_utc": datetime.now(UTC).isoformat(),
        "dataset": "Quick, Draw! simplified strokes, first recognized per category",
        "strokes_url": STROKES_URL,
        "per_category": args.per_category,
        "categories": list(args.categories),
        "seed": args.seed,
        "concurrency": image_concurrency(environ),
        "key_ids": key_ids,
        **served_weights_pins(backend, environ),
    }


def _run(args: argparse.Namespace, environ: Mapping[str, str]) -> dict[str, Any]:
    backend = live_backend(environ) if args.require_live else load_backend(environ)
    served_weights_pins(backend, environ)
    image_concurrency(environ)
    secret = load_vllm_settings(environ).api_key if backend == "vllm" else None
    doodles = [
        doodle
        for word in args.categories
        for doodle in fetch_doodles(word, args.per_category, cache_dir=args.cache_dir)
    ]
    requests = _requests(doodles, args.categories, args.seed)
    tree = _working_tree(environ)
    evaluated = snapshot_evaluated_inputs(
        prompts=(_prompt_spec(args.categories),),
        code_paths=_CODE_PATHS,
        fixture_paths={},
    )
    run, model, start = _judge(environ, requests, tree)
    identity = finalize_experiment_identity(
        run_start=start,
        evaluated=evaluated,
        arm_call_counts={"judgments": len(run.outcomes)},
    )
    receipt = build_doodle_receipt(
        run,
        backend=backend,
        model=model,
        pins=_pins(args, doodles, environ, backend),
        identity=identity.to_receipt_mapping(),
    )
    ensure_key_free(json.dumps(receipt), secrets=(secret,))
    write_receipt_exclusive(args.receipt, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run the doodle ``Choice`` evaluation and print the accuracy.

    Args:
        argv: CLI args; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every doodle ran; ``1`` when the receipt path exists, the
        backend or environment is refused, a cache miss fails to download,
        or the run stopped at a backend failure; ``2`` on a usage error,
        such as a repeated ``--categories`` name.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.per_category < 1:
            parser.error(f"--per-category must be at least 1: {args.per_category}")
    except SystemExit as exc:
        return _EXIT_OK if exc.code in (0, None) else _EXIT_USAGE
    if args.receipt.exists():
        print(f"refused: receipt path exists: {args.receipt}", file=sys.stderr)
        return _EXIT_REFUSED
    try:
        receipt = _run(args, dict(os.environ))
    except (ValueError, OSError, httpx.HTTPError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return _EXIT_REFUSED
    print(f"receipt {args.receipt}")
    print(f"accuracy {receipt['metrics']['accuracy']}")
    if receipt["stopped"] is not None:
        print(f"stopped {receipt['stopped']}", file=sys.stderr)
        return _EXIT_REFUSED
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
