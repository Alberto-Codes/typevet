# LOC gate

Kind: reference.

The line-of-code gate keeps modules small enough to decompose and keeps
function bodies short enough to read in one pass. Run it with:

```console
uv run python scripts/check_loc.py src
```

Pre-commit runs the same command on push-stage hooks. See the **loc** row in
[AGENTS.md](../../AGENTS.md).

## What gets counted

The script counts **code lines**: lines that carry at least one real token,
excluding comments and docstrings. Blank lines do not count. Module docstrings
and function docstrings do not count toward either cap.

For functions, only the **body** counts: decorators and the signature are
excluded. A nested function is measured on its own and its lines also count
toward the enclosing function.

## Limits (revision 1, issue #49)

| Scope | Limit | On exceed |
|---|---|---|
| Python module (file) | 300 code lines | **Fail** — a file at 301 or more must be split |
| Function or method body | 50 code lines | **Fail** — a body at 51 or more must be refactored |

The file cap is a **hard** limit at 300. This repo does **not** use a soft
320-line band; 301 code lines fails the gate.

The function cap is enforced the same way as the file cap: over the limit
prints a `FAIL` line and a non-zero exit code.

## Implementation

Logic lives in [scripts/check_loc.py](../../scripts/check_loc.py). Unit tests
in [tests/unit/test_check_loc.py](../../tests/unit/test_check_loc.py) cover
function-body enforcement.
