r"""Offline contact sheets of one synthetic check slice (#344).

The command makes the check slice for one seed and writes one contact sheet
per register row. Each sheet holds the 7 variants of that row with their
expected labels. The command makes no model call. Keep the output directory
outside the repository or under the git-ignored ``scratchpad/``.

Examples:
    ```console
    $ uv run python -m typevet_evals.cli.check_sheets \\
        --seed 1 --out scratchpad/check_sheets_seed1
    scratchpad/check_sheets_seed1/seed1_row00.png
    ...
    ```

See Also:
    - [typevet_evals.check_match.cases][]: ``check_cases`` and the seed
    - [typevet_evals.check_match.render][]: ``write_contact_sheet``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typevet_evals.check_match import (
    DEFAULT_SEED,
    ROW_COUNT,
    CheckVariant,
    check_cases,
    write_contact_sheet,
)

_EXIT_OK = 0
_EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write one contact sheet per register row of the synthetic check "
            "slice for one seed. No model call."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Generator seed (default 0)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=ROW_COUNT,
        help=f"Register rows, 1 to {ROW_COUNT} (default {ROW_COUNT})",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Directory for the PNG sheets"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the contact sheets and print one path per line.

    Args:
        argv: CLI args; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every sheet is written; ``2`` on a usage error.
    """
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return _EXIT_OK if exc.code in (0, None) else _EXIT_USAGE
    if not 1 <= args.rows <= ROW_COUNT:
        print(f"error: --rows must be 1 to {ROW_COUNT}: {args.rows}", file=sys.stderr)
        return _EXIT_USAGE
    if args.seed < 0:
        print(f"error: --seed must be non-negative: {args.seed}", file=sys.stderr)
        return _EXIT_USAGE
    cases = check_cases(args.seed, args.rows)
    per_row = len(CheckVariant)
    for row in range(args.rows):
        target = args.out / f"seed{args.seed}_row{row:02d}.png"
        row_cases = cases[row * per_row : (row + 1) * per_row]
        print(write_contact_sheet(row_cases, target, columns=per_row))
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
