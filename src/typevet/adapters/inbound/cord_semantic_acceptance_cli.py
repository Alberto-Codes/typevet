r"""Operator CLI for offline CORD combined receipt semantic acceptance (#184).

Examples:
    ```console
    $ uv run python -m typevet.adapters.inbound.cord_semantic_acceptance_cli \\
        tests/fixtures/cord/semantic_acceptance/labeled_synthetic_pass.json
    | check | bound | limit | measured | n | status |
    ...
    accepted: true
    ```

See Also:
    - [typevet.evaluation.cord_semantic_acceptance][]: ``accept_combined_receipt``
    - [typevet.evaluation.cord_semantic_acceptance_report][]: table formatting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from typevet.evaluation.cord_semantic_acceptance_report import (
    evaluate_combined_receipt,
    format_semantic_acceptance_report,
)

_EXIT_ACCEPT = 0
_EXIT_REJECT = 1
_EXIT_USAGE_OR_MALFORMED = 2


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Evaluate one saved CORD combined-modality receipt against the "
            "frozen #161 revision 1 semantic floors. Exit 0 only when accepted."
        ),
    )


def _load_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"receipt path is not a file: {path}"
        raise ValueError(msg)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read receipt: {path}"
        raise ValueError(msg) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"receipt is not valid JSON: {path}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        msg = "receipt root must be a JSON object"
        raise TypeError(msg)
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Load a receipt path, print the floor table, return a process exit code.

    Args:
        argv: CLI args; defaults to ``sys.argv[1:]`` (one receipt path).

    Returns:
        ``0`` when ``accepted`` is true, ``1`` when checks fail, ``2`` on
        usage errors, malformed receipts, or ``ValueError`` / ``TypeError`` from
        parsing.
    """
    parser = _build_parser()
    parser.add_argument(
        "receipt",
        type=Path,
        help="Path to a saved combined CORD outcome JSON file",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code in (0, None):
            return _EXIT_ACCEPT
        return (
            _EXIT_USAGE_OR_MALFORMED
            if code == _EXIT_USAGE_OR_MALFORMED
            else _EXIT_REJECT
        )
    try:
        receipt = _load_receipt(args.receipt)
        outcome = evaluate_combined_receipt(receipt)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USAGE_OR_MALFORMED
    print(format_semantic_acceptance_report(outcome))
    return _EXIT_ACCEPT if outcome.accepted else _EXIT_REJECT


if __name__ == "__main__":
    raise SystemExit(main())
