"""Check owned Markdown against a conservative terminology replacement map."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
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

CONFIG = Path(__file__).with_name("terminology.toml")
GLOSSARY = Path(__file__).resolve().parents[1] / "docs/reference/glossary.md"
Rule = tuple[re.Pattern[str], str]


def in_scope(path: Path) -> bool:
    """Select top-level policy files and Markdown throughout docs."""
    return path.suffix == ".md" and (
        path.as_posix() in {"README.md", "CLAUDE.md", "AGENTS.md"}
        or path.parts[:1] == ("docs",)
    )


def _term(value: object) -> str:
    """Validate a nonempty term with single spaces and no control characters."""
    if not isinstance(value, str) or not value or value != " ".join(value.split()):
        raise ValueError("terms must be nonempty strings with single spaces")
    if not value.isprintable():
        raise ValueError("terms cannot contain control characters")
    return value


def _entry(value: object) -> tuple[str, str]:
    """Validate one forbidden-to-preferred mapping entry."""
    if not isinstance(value, dict) or set(value) != {"forbidden", "preferred"}:
        raise ValueError("each term requires only forbidden and preferred fields")
    return _term(value["forbidden"]), _term(value["preferred"])


def _glossary_terms(path: Path) -> set[str]:
    """Read bold glossary entries and Markdown headings as canonical terms."""
    text = read_markdown(path)
    patterns = (r"^\*\*([^*\n]+)\.\*\*", r"^#{1,6} +(.+?)(?: +#+)?$")
    return {
        match[1].strip().casefold()
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.MULTILINE)
    }


def load_rules(config: Path, glossary: Path) -> list[Rule]:
    """Load and validate all mappings before any prose is checked.

    Reject duplicate or overlapping forbidden phrases. A forbidden phrase
    cannot occur within a canonical glossary term.
    """
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    entries = data.get("terms")
    if set(data) != {"terms"} or not isinstance(entries, list) or not entries:
        raise ValueError("configuration requires a nonempty terms array")
    canonical = _glossary_terms(glossary)
    rules: list[Rule] = []
    sources: list[str] = []
    for entry in entries:
        forbidden, preferred = _entry(entry)
        pattern = re.compile(
            r"(?<!\w)" + re.escape(forbidden) + r"(?!\w)", re.IGNORECASE
        )
        if preferred.casefold() not in canonical:
            raise ValueError(f"preferred term is absent from glossary: {preferred}")
        if any(pattern.search(term) for term in canonical):
            raise ValueError(f"forbidden phrase collides with glossary: {forbidden}")
        if any(pattern.search(source) for source in sources) or any(
            previous.search(forbidden) for previous, _ in rules
        ):
            raise ValueError(f"duplicate or overlapping forbidden phrase: {forbidden}")
        sources.append(forbidden)
        rules.append((pattern, preferred))
    return rules


def _findings(text: str, rules: list[Rule]) -> list[str]:
    """Report each forbidden phrase with its canonical replacement."""
    return [
        f"use '{preferred}' instead of '{match[0]}'"
        for pattern, preferred in rules
        for match in pattern.finditer(text)
    ]


def main(args: list[str]) -> int:
    """Check explicit files, a saved patch, or current owned sentences.

    Return 1 for findings and 2 for invalid input or configuration.
    Report selected file and paragraph counts, including zero ownership.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--diff", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    options = parser.parse_args(args)
    if options.diff and options.paths:
        print("ERROR: --diff cannot be combined with explicit files", file=sys.stderr)
        return 2
    try:
        rules = load_rules(options.config, GLOSSARY)
        if options.paths:
            selection = {Path(path).absolute(): None for path in options.paths}
        elif options.diff:
            selection = patch_selection(
                options.diff.read_text(encoding="utf-8"), repository_root(), in_scope
            )
        else:
            selection = changed_selection(repository_root(), in_scope)
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
                for finding in _findings(prose, rules):
                    print(f"{path}:{start}: {finding}")
                    failures += 1
        print(f"checked {len(selection)} files, {count} owned paragraphs")
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
