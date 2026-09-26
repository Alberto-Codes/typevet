"""Extract lightweight Markdown paragraphs and select owned prose.

This scanner preserves source line spans. It is not a CommonMark parser.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

Paragraph = tuple[int, int, str]
Selection = dict[Path, set[int] | None]


def _omit_literal(match: re.Match[str]) -> str:
    """Keep sentence punctuation after a URL or path."""
    return " " + match[0][len(match[0].rstrip(".?!,;:")) :]


def _visible(text: str) -> str:
    """Remove inline literals and destinations, retaining visible link labels."""
    text = re.sub(r"(`+).*?\1", " ", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"(?:https?://|www\.)[^\s<>]+", _omit_literal, text)
    text = re.sub(r"(?<!\w)(?:[\w.-]+/|[./~]+/)[^\s<>|]*", _omit_literal, text)
    text = re.sub(r"\b[\w-]+\.(?:md|py|json|toml|yaml|yml|txt)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _lines(text: str) -> list[str]:
    """Blank block exclusions without changing line numbers."""
    lines = text.splitlines()
    fence = ""
    front = bool(lines and lines[0].strip() == "---")
    for index, line in enumerate(lines):
        stripped = line.strip()
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if front:
            lines[index] = ""
            if index and stripped in {"---", "..."}:
                front = False
        elif fence:
            lines[index] = ""
            if re.fullmatch(
                re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*", stripped
            ):
                fence = ""
        elif marker:
            fence = marker[1]
            lines[index] = ""
        elif line.startswith(("    ", "\t")) or re.match(r"^ {0,3}\[[^]]+\]:", line):
            lines[index] = ""
    text = "\n".join(re.sub(r"(`+).*?\1", " ", line) for line in lines)
    text = re.sub(
        r"<!--.*?-->", lambda m: "\n" * m[0].count("\n"), text, flags=re.DOTALL
    )
    if "<!--" in text:
        raise ValueError("unclosed HTML comment")
    return text.split("\n")


def _quoted_and_headers(lines: list[str]) -> list[str]:
    """Blank cited quote blocks and table headers."""
    index = 0
    while index < len(lines):
        if lines[index].lstrip().startswith(">"):
            end = index + 1
            while end < len(lines) and lines[end].lstrip().startswith(">"):
                end += 1
            quote = " ".join(lines[index:end])
            if re.search(r"https?://|\[[^]]+\]\[[^]]+\]", quote):
                lines[index:end] = [""] * (end - index)
            index = end
        else:
            index += 1
    for index, line in enumerate(lines):
        if "|" in line and re.fullmatch(
            r"\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*", line
        ):
            lines[index] = ""
            if index:
                lines[index - 1] = ""
    return lines


def _paragraphs(text: str) -> list[Paragraph]:
    """Return visible paragraphs with inclusive, one-based line spans.

    Lists, headings and table rows start separate paragraphs. Wrapped prose
    remains one paragraph so ownership includes the complete sentence.
    """
    result: list[Paragraph] = []
    pending: list[str] = []
    start = 1
    lines = _quoted_and_headers(_lines(text))
    for number, line in enumerate([*lines, ""], 1):
        boundary = bool(re.match(r"^\s*(?:#{1,6}\s|[-+*]\s|\d+[.)]\s)", line))
        table = "|" in line
        if pending and (not line.strip() or boundary or table):
            result.append((start, number - 1, "\n".join(pending)))
            pending = []
        if line.strip():
            if not pending:
                start = number
            content = re.sub(r"^\s*(?:#{1,6}\s+|[-+*]\s+|\d+[.)]\s+|>\s*)", "", line)
            if re.match(r"^\s*#{1,6}\s", line):
                result.append((number, number, content))
            elif table:
                result.extend(
                    (number, number, cell)
                    for cell in content.strip("|").split("|")
                    if cell.strip()
                )
            else:
                pending.append(content)
    return [item for item in result if item[2]]


def paragraphs(text: str) -> list[Paragraph]:
    """Return visible paragraphs with inclusive source line spans."""
    return [
        (start, end, _visible(prose))
        for start, end, prose in _paragraphs(text)
        if _visible(prose)
    ]


def sentences(text: str) -> list[Paragraph]:
    """Return complete visible sentences with their source line spans.

    Sentence punctuation and paragraph ends delimit sentences. A wrapped
    sentence includes all its lines, without owning adjacent sentences.
    """
    result: list[Paragraph] = []
    for start, _, prose in _paragraphs(text):
        visible = "\n".join(_visible(line) for line in prose.split("\n"))
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", visible):
            fragment = match[0]
            if not fragment.strip():
                continue
            first = match.start() + len(fragment) - len(fragment.lstrip())
            last = match.end() - len(fragment) + len(fragment.rstrip())
            result.append(
                (
                    start + visible.count("\n", 0, first),
                    start + visible.count("\n", 0, last),
                    " ".join(fragment.split()),
                )
            )
    return result


def in_scope(path: Path) -> bool:
    """Identify strict default Markdown paths relative to the repository."""
    return path.suffix == ".md" and (
        path.as_posix() in {"README.md", "CLAUDE.md", "AGENTS.md"}
        or (
            path.parts[:2]
            in {("docs", "reference"), ("docs", "how-to"), ("docs", "maintainers")}
        )
    )


def read_markdown(path: Path) -> str:
    """Read one Markdown file, rejecting invalid explicit paths."""
    if path.suffix != ".md" or not path.is_file():
        raise ValueError(f"expected a Markdown file: {path}")
    return path.read_text(encoding="utf-8")


def _patch_path(value: str, prefix: str) -> Path | None:
    """Decode supported Git paths, rejecting quoted or unsafe paths."""
    if value == "/dev/null":
        return None
    if not value.startswith(prefix) or any(char in value for char in '\t"\\'):
        raise ValueError("unsupported patch path")
    path = Path(value[len(prefix) :])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe patch path")
    return path


def _hunk(
    lines: list[str], index: int, current: list[str] | None
) -> tuple[int, set[int]]:
    """Validate a zero-context hunk and match added lines to current text."""
    match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*", lines[index])
    if not match:
        raise ValueError("malformed patch hunk")
    old_left = int(match[2] or "1")
    new_left = int(match[4] or "1")
    number = int(match[3])
    selected: set[int] = set()
    index += 1
    while index < len(lines) and (old_left or new_left):
        line = lines[index]
        if line.startswith("-") and old_left:
            old_left -= 1
        elif line.startswith("+") and new_left:
            if current is not None and (
                number < 1 or number > len(current) or current[number - 1] != line[1:]
            ):
                raise ValueError("patch does not match current file")
            selected.add(number)
            number += 1
            new_left -= 1
        elif line != "\\ No newline at end of file":
            raise ValueError("expected a zero-context patch")
        index += 1
    if old_left or new_left:
        raise ValueError("truncated patch hunk")
    if index < len(lines) and lines[index] == "\\ No newline at end of file":
        index += 1
    return index, selected


def _header_paths(line: str) -> tuple[Path, Path]:
    """Read safe old and new paths from the Git section header."""
    match = re.fullmatch(r"diff --git a/(.+) b/(.+)", line)
    if not match:
        raise ValueError("unsupported patch header")
    old = _patch_path("a/" + match[1], "a/")
    new = _patch_path("b/" + match[2], "b/")
    if old is None or new is None:
        raise ValueError("missing patch path")
    return old, new


def _metadata_only(lines: list[str], path: Path) -> None:
    """Accept mode-only or empty-file metadata, validating the current file."""
    metadata = "\n".join(lines[1:])
    if re.fullmatch(r"old mode [0-7]{6}\nnew mode [0-7]{6}", metadata):
        read_markdown(path)
    elif re.fullmatch(r"new file mode [0-7]{6}\nindex 0+\.[.][a-f0-9]+", metadata):
        if read_markdown(path):
            raise ValueError("empty-file patch does not match current file")
    elif re.fullmatch(r"deleted file mode [0-7]{6}\nindex [a-f0-9]+\.[.]0+", metadata):
        if path.exists():
            raise ValueError("deleted patch path still exists")
    else:
        raise ValueError("unsupported or malformed patch section")


def _section(
    lines: list[str], root: Path, scope: Callable[[Path], bool]
) -> tuple[Path | None, set[int]]:
    """Validate owned Git diff sections and ignore unrelated file contents."""
    header_old, header_new = _header_paths(lines[0])
    owned = scope(header_old) or scope(header_new)
    index = 1
    while index < len(lines) and re.fullmatch(
        r"(?:index [a-f0-9]+\.\.[a-f0-9]+(?: [0-7]{6})?"
        r"|(?:new file|deleted file|old|new) mode [0-7]{6})",
        lines[index],
    ):
        index += 1
    if index == len(lines) or not lines[index].startswith("--- "):
        if owned:
            _metadata_only(lines, root / header_new)
        return None, set()
    if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
        raise ValueError("malformed patch file headers")
    old = _patch_path(lines[index][4:], "a/")
    new = _patch_path(lines[index + 1][4:], "b/")
    if old not in {None, header_old} or new not in {None, header_new}:
        raise ValueError("patch header does not match its paths")
    if not owned:
        return None, set()
    if header_old != header_new or (old is None and new is None):
        raise ValueError("unsupported patch rename")
    current = read_markdown(root / header_new).splitlines() if new else None
    if new is None and (root / header_old).exists():
        raise ValueError("deleted patch path still exists")
    index += 2
    selected: set[int] = set()
    if index == len(lines):
        raise ValueError("patch section has no hunks")
    while index < len(lines):
        index, added = _hunk(lines, index, current)
        selected.update(added)
    if new is None and selected:
        raise ValueError("deleted patch contains added lines")
    return (root / header_new if new else None), selected


def patch_selection(
    text: str, root: Path, scope: Callable[[Path], bool] = in_scope
) -> Selection:
    """Select added lines from a supported zero-context Git patch.

    Invalid owned patches raise ValueError. Non-owned file contents are
    ignored after path validation; metadata-only changes select no prose.
    """
    if not text.strip():
        return {}
    lines = text.splitlines()
    starts = [
        index for index, line in enumerate(lines) if line.startswith("diff --git ")
    ]
    if not starts or starts[0] != 0:
        raise ValueError("expected a Git patch")
    result: Selection = {}
    for start, end in zip(starts, [*starts[1:], len(lines)], strict=True):
        path, added = _section(lines[start:end], root, scope)
        if path is not None and added:
            previous = result.setdefault(path, set())
            if previous is not None:
                previous.update(added)
    return result


def repository_root() -> Path:
    """Resolve the current Git repository root."""
    result = subprocess.run(
        ["/usr/bin/git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def changed_selection(
    root: Path, scope: Callable[[Path], bool] = in_scope
) -> Selection:
    """Select tracked changes against HEAD and untracked strict Markdown."""
    diff = subprocess.run(
        [
            "/usr/bin/git",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--unified=0",
            "HEAD",
            "--",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    result = patch_selection(diff.stdout, root, scope)
    untracked = subprocess.run(
        ["/usr/bin/git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    for name in untracked.stdout.split("\0"):
        if name and scope(Path(name)):
            result[root / name] = None
    return result
