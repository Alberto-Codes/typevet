"""Check the sentence limit and banned adjectives in owned Markdown prose."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

if __package__:
    from scripts.markdown_prose import (
        changed_selection,
        paragraphs,
        patch_selection,
        read_markdown,
        repository_root,
        sentences,
    )
else:
    from markdown_prose import (
        changed_selection,
        paragraphs,
        patch_selection,
        read_markdown,
        repository_root,
        sentences,
    )

LIMIT = 20
WORDS = re.compile(r"[^\W_]+(?:['\u2019\-][^\W_]+)*", re.UNICODE)
BANNED = re.compile(
    r"\b(?:seamless|robust|powerful|blazing|cutting-edge)\b", re.IGNORECASE
)


def _findings(text: str) -> list[str]:
    """Report sentence length and banned whole-word adjectives."""
    findings = [f"banned adjective: {match[0]}" for match in BANNED.finditer(text)]
    count = len(WORDS.findall(text))
    if count > LIMIT:
        findings.append(f"{count} words (limit {LIMIT})")
    return findings


def main(args: list[str]) -> int:
    """Check explicit files, a saved patch, or current owned paragraphs.

    Return 1 for findings and 2 for invalid input. Print ownership counts
    even when no paragraphs are selected.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--diff", type=Path)
    options = parser.parse_args(args)
    if options.diff and options.paths:
        print("ERROR: --diff cannot be combined with explicit files", file=sys.stderr)
        return 2
    try:
        if options.paths:
            selection = {Path(path).absolute(): None for path in options.paths}
        elif options.diff:
            selection = patch_selection(
                options.diff.read_text(encoding="utf-8"), repository_root()
            )
        else:
            selection = changed_selection(repository_root())
        failures = 0
        count = 0
        for path, owned in selection.items():
            text = read_markdown(path)
            selected = [
                sentence
                for sentence in sentences(text)
                if owned is None
                or not owned.isdisjoint(range(sentence[0], sentence[1] + 1))
            ]
            count += sum(
                any(first <= end and last >= start for first, last, _ in selected)
                for start, end, _ in paragraphs(text)
            )
            for start, _, prose in selected:
                for finding in _findings(prose):
                    print(f"{path}:{start}: {finding}")
                    failures += 1
        print(f"checked {len(selection)} files, {count} owned paragraphs")
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
