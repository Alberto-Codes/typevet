"""Guardrails: collections NBA partner paths must not ship in public typevet.

The finvet collections NBA split is partner-only and git-ignored in finvet.
typevet eval bundles and wheels must never vendor those paths or jsonl.

Examples:
    Scan a tree before release:

    ```python
    from pathlib import Path

    from typevet.evaluation.datasets.partner_guard import scan_tree_paths

    assert scan_tree_paths(Path(".")) == []
    ```

See Also:
    - [typevet][]: Public package this guard protects
"""

from __future__ import annotations

import subprocess
from pathlib import Path

FORBIDDEN_PATH_MARKERS: tuple[str, ...] = (
    "collections_nba",
    "data/collections_nba",
    "collections/nba",
)

FORBIDDEN_IMPORT_MARKERS: tuple[str, ...] = ("finvet.data.collections_nba",)

# Files that may mention forbidden markers when describing the policy.
CONTENT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "docs/reference/eval-partner-data-policy.md",
        "src/typevet/evaluation/datasets/partner_guard.py",
        "tests/unit/test_eval_partner_guard.py",
    }
)

PACKAGING_FILES: frozenset[str] = frozenset({"pyproject.toml", "MANIFEST.in"})


def normalize_posix(path: str) -> str:
    """Return a lower-case forward-slash path for comparisons.

    Returns:
        The normalized path string.
    """
    return path.replace("\\", "/").lower()


def path_is_forbidden(relative_path: str) -> bool:
    """Return True when a repo-relative path looks like partner NBA data.

    Returns:
        True when any forbidden path marker appears in the path.
    """
    norm = normalize_posix(relative_path)
    return any(marker in norm for marker in FORBIDDEN_PATH_MARKERS)


def packaging_line_is_forbidden(line: str) -> bool:
    """Return True when a packaging config line would ship partner paths.

    Returns:
        True for non-comment lines that mention forbidden path markers.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    lower = stripped.lower()
    return any(marker in lower for marker in FORBIDDEN_PATH_MARKERS)


def scan_tree_paths(root: Path) -> list[str]:
    """List repo-relative file paths under ``root`` that violate path policy.

    Returns:
        Sorted relative paths that match forbidden markers.
    """
    if not root.is_dir():
        return []
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path_is_forbidden(rel):
            hits.append(rel)
    return sorted(hits)


def scan_tracked_paths(repo_root: Path) -> list[str]:
    """List git-tracked paths that violate path policy.

    Returns:
        Sorted tracked paths that match forbidden markers.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    hits: list[str] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode()
        if path_is_forbidden(rel):
            hits.append(rel)
    return sorted(hits)


def scan_tracked_content(repo_root: Path) -> list[str]:
    """List tracked files whose text mentions forbidden markers outside allowlist.

    Returns:
        Sorted repo-relative paths whose UTF-8 text violates content policy.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    hits: list[str] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode()
        if rel in CONTENT_ALLOWLIST:
            continue
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lower = text.lower()
        path_hit = any(marker in lower for marker in FORBIDDEN_PATH_MARKERS)
        import_hit = rel.startswith("src/") and any(
            marker in lower for marker in FORBIDDEN_IMPORT_MARKERS
        )
        if path_hit or import_hit:
            hits.append(rel)
    return sorted(hits)


def scan_packaging_config(repo_root: Path) -> list[str]:
    """List packaging files that mention forbidden path markers on active lines.

    Returns:
        Sorted ``file:line:text`` entries for each violating packaging line.
    """
    hits: list[str] = []
    for name in PACKAGING_FILES:
        path = repo_root / name
        if not path.is_file():
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if packaging_line_is_forbidden(line):
                hits.append(f"{name}:{index}:{line.strip()}")
    return sorted(hits)
