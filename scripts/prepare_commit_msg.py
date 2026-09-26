"""Strip harness ``Co-Authored-By`` lines before commit-msg validation (#85).

Cursor and other harnesses auto-append model co-author trailers. The
``prepare-commit-msg`` hook removes known harness identities so the stored
commit message matches repo policy; ``check_commit_msg`` still rejects any
forbidden trailer that remains.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _strip_harness_coauthors_loader() -> Callable[[str], str]:
    module_path = _REPO_ROOT / "scripts" / "check_commit_msg.py"
    spec = importlib.util.spec_from_file_location(
        "typevet_check_commit_msg",
        module_path,
    )
    if spec is None or spec.loader is None:
        msg = f"cannot load commit-msg helpers from {module_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strip: Callable[[str], str] = module.strip_harness_coauthors
    return strip


def main(argv: list[str] | None = None) -> int:
    """Rewrite each commit message file with harness co-authors removed.

    Args:
        argv: Message file paths from pre-commit. None reads ``sys.argv``.

    Returns:
        Process exit code, always 0 when paths are processed.
    """
    strip_harness_coauthors = _strip_harness_coauthors_loader()
    args = list(argv if argv is not None else sys.argv[1:])
    for path in (Path(arg) for arg in args):
        if not path.is_file():
            continue
        path.write_text(
            strip_harness_coauthors(path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
