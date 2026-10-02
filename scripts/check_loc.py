"""File-size gate: fail when a module exceeds the decomposition limit.

Counts *code* lines — lines carrying at least one real token, excluding
comments and docstrings — so docvet-mandated documentation never pushes a
file over the limit. The limit is 300 code lines: a module at 300
passes, and a module at 301 or more fails the gate. Anything larger gets
decomposed, not excused.

Functions are measured the same way, over the body only: decorators and
the signature do not count, and a nested function counts toward its
parent. A function past 50 code lines fails the gate.

Examples:
    Run against the source tree:

    ```console
    $ uv run python scripts/check_loc.py src
    checked 14 files
    ```

    Count the body lines of every function in one file:

    ```python
    from pathlib import Path

    from scripts.check_loc import count_function_lines

    for qualname, lines in count_function_lines(Path("scripts/check_loc.py")):
        print(qualname, lines)
    ```

See Also:
    - [typevet][]: Package whose modules the cap applies to.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

# --8<-- [start:limits]
LIMIT = 300
FUNCTION_LIMIT = 50
# --8<-- [end:limits]

_SKIP_TOKENS = frozenset(
    {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.ENDMARKER,
        tokenize.ENCODING,
    }
)


def _docstring_lines(tree: ast.Module) -> set[int]:
    """Collect the line numbers occupied by docstrings.

    Args:
        tree: Parsed module AST.

    Returns:
        All 1-based line numbers inside module, class, or function
        docstrings.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
            and body[0].end_lineno is not None
        ):
            lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return lines


def _code_lines(text: str) -> set[int]:
    """Collect the line numbers that carry code.

    A line carries code when it holds at least one token that is not a
    comment, and it is not part of a docstring.

    Args:
        text: Python source text.

    Returns:
        All 1-based code line numbers.
    """
    doc_lines = _docstring_lines(ast.parse(text))
    token_lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in _SKIP_TOKENS:
            continue
        token_lines.update(range(tok.start[0], tok.end[0] + 1))
    return token_lines - doc_lines


def count_code_lines(path: Path) -> int:
    """Count lines of actual code in a Python file.

    A line counts when it carries at least one token that is not a
    comment, and it is not part of a docstring.

    Args:
        path: The Python file to measure.

    Returns:
        The number of code lines.
    """
    return len(_code_lines(path.read_text(encoding="utf-8")))


def count_function_lines(path: Path) -> list[tuple[str, int]]:
    """Count the body code lines of every function and method in a file.

    A body spans its first statement to the function's last line, so
    decorators and the signature are excluded. Docstring, comment and
    blank lines do not count. A nested function is listed on its own and
    also counts toward its parent.

    Args:
        path: The Python file to measure.

    Returns:
        ``(qualname, code lines)`` pairs in source order, where the
        qualname joins enclosing class and function names with dots.
    """
    text = path.read_text(encoding="utf-8")
    code = _code_lines(text)
    results: list[tuple[str, int]] = []
    stack: list[tuple[ast.AST, str]] = [(ast.parse(text), "")]
    while stack:
        node, prefix = stack.pop()
        children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            name = getattr(child, "name", "")
            qualname = f"{prefix}{name}" if name else prefix.rstrip(".")
            stack.append((child, f"{qualname}." if qualname else ""))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.body[-1].lineno
            span = range(node.body[0].lineno, end + 1)
            results.append((prefix.rstrip("."), len(code.intersection(span))))
    return results


def _function_failures(path: Path) -> list[tuple[str, int]]:
    """List every function in `path` past the function limit.

    Args:
        path: The Python file to measure.

    Returns:
        ``(qualname, code lines)`` pairs for functions over the limit.
    """
    return [
        (qualname, n)
        for qualname, n in count_function_lines(path)
        if n > FUNCTION_LIMIT
    ]


def main(roots: list[str]) -> int:
    """Check every Python file under the given roots.

    Prints a ``FAIL`` line for each file or function past its limit.

    Args:
        roots: Directories to scan (defaults to ``src`` when empty).

    Returns:
        Process exit code: 1 if any file or function exceeds its limit,
        else 0.
    """
    failures = 0
    checked = 0
    for root in roots or ["src"]:
        paths = sorted(Path(root).rglob("*.py"))
        if not paths:
            # A gate that scans nothing must fail loudly, not pass
            # silently — a renamed root would otherwise disable it.
            print(f"FAIL {root}: no Python files found")
            failures += 1
            continue
        for path in paths:
            checked += 1
            n = count_code_lines(path)
            for qualname, fn_lines in _function_failures(path):
                print(
                    f"FAIL {path}:{qualname}: {fn_lines} code lines "
                    f"(function limit {FUNCTION_LIMIT})"
                )
                failures += 1
            if n > LIMIT:
                print(f"FAIL {path}: {n} code lines (limit {LIMIT})")
                failures += 1
    print(f"checked {checked} files")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
