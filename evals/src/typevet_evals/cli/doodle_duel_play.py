r"""Play a doodle duel against a doodle receipt, with no model call (#412).

The command replays the rows of a doodle receipt in order, one round per
row. Each round writes the drawing to ``round-001.png`` (and so on) in the
session directory, takes a pick and a confidence from 50 to 100, then shows
the answer, the person's pick and the model's pick. Both players get the
same pick Brier score; lower Brier is better. The command prints accuracy,
mean Brier, the always-50% baseline, the model's multi-class Brier and a
reliability table for both players, then writes a duel receipt with no
image bytes and no file paths of PNGs.

Answers come from the terminal, or from ``--answers``: a JSON list of
``{"key_id", "label", "confidence"}``, one per round in round order. The
drawings come from the Quick, Draw! cache; a cache miss downloads the same
prefix from Google's public bucket. Exit codes: ``0`` when every round
played and the receipt was written; ``1`` on a refusal (an existing duel
receipt, a session directory that is not empty, a bad source receipt or
answers file, a missing drawing, a failed download, end of input); ``2`` on
a usage error. A refusal writes no duel receipt.

Examples:
    ```console
    $ uv run python -m typevet_evals.cli.doodle_duel_play \\
        --source-receipt evals/fixtures/quickdraw/receipts/doodle_duel_llama_cpp_receipt.json \\
        --duel-receipt duel.json --answers answers.json
    ```

See Also:
    - [typevet_evals.doodle_duel.duel][]: scores, bins and the receipt
    - [typevet_evals.doodle_duel.duel_session][]: rounds, prompts and drawings
    - [typevet_evals.cli.doodle_duel][]: the run that writes the source receipt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from typevet_evals.doodle_duel import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    build_duel_receipt,
    doodles_for_rows,
    file_answers,
    interactive_answers,
    load_source_receipt,
    parse_answers,
    play_duel,
)
from typevet_evals.experiment_identity import write_receipt_exclusive

_EXIT_OK = 0
_EXIT_REFUSED = 1
_EXIT_USAGE = 2
_LOWER = "lower Brier is better"
_INTRO = (
    "Each round: name the drawing, then say how sure you are, "
    f"{CONFIDENCE_MIN} to {CONFIDENCE_MAX}. "
    f"{CONFIDENCE_MIN} means a coin flip, {CONFIDENCE_MAX} means certain. "
    "Your number is the percent chance your pick is right."
)
_DEFINITIONS = (
    (
        "Brier: squared distance between your confidence and what happened; "
        "0 is perfect, 1 is worst"
    ),
    "baseline: what always saying 50 percent would score",
    "gap: mean confidence minus accuracy; positive means overconfident",
)
_ZERO = "0.000"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a doodle duel against a doodle receipt.",
    )
    parser.add_argument(
        "--source-receipt", type=Path, required=True, help="Doodle receipt"
    )
    parser.add_argument("--duel-receipt", type=Path, required=True, help="New file")
    parser.add_argument("--answers", type=Path, default=None, help="Answers file")
    parser.add_argument("--rounds", type=int, default=None, help="Default all rows")
    parser.add_argument("--session-dir", type=Path, default=None, help="PNG dir")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache dir")
    return parser


def _session_dir(args: argparse.Namespace) -> Path:
    if args.session_dir is not None:
        return args.session_dir
    duel: Path = args.duel_receipt
    return duel.with_name(f"{duel.stem}-rounds")


def _refusal(args: argparse.Namespace, session: Path) -> str | None:
    if args.duel_receipt.exists():
        return f"duel receipt path exists: {args.duel_receipt}"
    if session.exists() and (not session.is_dir() or any(session.iterdir())):
        return f"session directory is not empty: {session}"
    return None


def _format(value: float | None) -> str:
    if value is None:
        return "-"
    text = f"{value:.3f}"
    return _ZERO if text == f"-{_ZERO}" else text


def _print_summary(receipt: Mapping[str, Any], write: Callable[[str], object]) -> None:
    scores = receipt["scores"]
    for who, name in (("player", "you"), ("model", "model")):
        s = scores[who]
        write(
            f"{name}: accuracy {s['accuracy']:.3f}, "
            f"mean brier {s['mean_brier']:.3f} ({_LOWER})"
        )
    write(f"always-50% baseline brier {scores['baseline_brier']:.3f} ({_LOWER})")
    multiclass = scores["model"]["multiclass_brier"]
    write(f"model multi-class brier {multiclass:.3f} (advanced; {_LOWER})")
    for line in _DEFINITIONS:
        write(line)
    for who, name in (("player", "you"), ("model", "model")):
        write(f"reliability, {name} (gap = mean confidence - accuracy):")
        write("bin     count  confidence  accuracy  gap")
        for row in receipt["reliability"][who]:
            write(
                f"{row['bin']:<7} {row['count']:>5}  "
                f"{_format(row['mean_confidence']):>10}  "
                f"{_format(row['accuracy']):>8}  {_format(row['gap']):>6}"
            )


def _play(
    args: argparse.Namespace,
    session: Path,
    read_line: Callable[[str], str],
    write: Callable[[str], object],
) -> dict[str, Any]:
    source = load_source_receipt(args.source_receipt)
    categories: Sequence[str] = source["pins"]["categories"]
    rows = source["rows"][: args.rounds]
    if args.answers is None:
        answer = interactive_answers(read_line, write, categories)
    else:
        raw = json.loads(args.answers.read_text(encoding="utf-8"))
        keys = [str(row["key_id"]) for row in rows]
        answer = file_answers(parse_answers(raw, keys, categories))
    doodles = doodles_for_rows(source, rows, cache_dir=args.cache_dir)
    if args.answers is None:
        write(_INTRO)
    records = play_duel(
        rows, doodles, categories, answer, session_dir=session, write=write
    )
    receipt = build_duel_receipt(
        records,
        source_rows=rows,
        categories=categories,
        input_mode="interactive" if args.answers is None else "answers_file",
        source_receipt={
            "path": str(args.source_receipt),
            "sha256": hashlib.sha256(args.source_receipt.read_bytes()).hexdigest(),
            "backend": str(source.get("backend")),
            "model": str(source.get("model")),
        },
        per_category=int(source["pins"]["per_category"]),
    )
    write_receipt_exclusive(args.duel_receipt, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Play the duel, print the scores and write the duel receipt.

    Args:
        argv: CLI args; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every round played and the receipt was written; ``1`` on
        a refusal; ``2`` on a usage error.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.rounds is not None and args.rounds < 1:
            parser.error(f"--rounds must be at least 1: {args.rounds}")
    except SystemExit as exc:
        return _EXIT_OK if exc.code in (0, None) else _EXIT_USAGE
    session = _session_dir(args)
    refusal = _refusal(args, session)
    if refusal is not None:
        print(f"refused: {refusal}", file=sys.stderr)
        return _EXIT_REFUSED
    try:
        receipt = _play(args, session, input, print)
    except EOFError:
        print("refused: input ended before the last round", file=sys.stderr)
        return _EXIT_REFUSED
    except (ValueError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return _EXIT_REFUSED
    print(f"duel receipt {args.duel_receipt}")
    print(f"rounds {receipt['rounds']}")
    _print_summary(receipt, print)
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
