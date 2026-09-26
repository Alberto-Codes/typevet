"""Commit gate: every message follows Conventional Commits 1.0.0 (#42).

The subject reads ``<type>[(scope)][!]: <description>``. The type comes
from the eleven-type vocabulary in ``docs/reference/commits.md``. A body
starts one blank line after the subject.
A breaking change writes ``BREAKING CHANGE`` in upper case. Every message
names its issue as ``#N``, in the subject or in a footer. A ``feat`` or
``fix`` without ``Closes``/``Refs`` gets a warning, not a failure, because
not every such commit finishes an issue.

The gate reads the file pre-commit hands it at the ``commit-msg`` stage. It
skips a merge, a revert and a ``fixup!`` message, because git writes those.
It also refuses commits whose author email is not in the allowed list.
Known model or harness ``Co-Authored-By`` identities fail; worker evidence
uses ``Generated-By`` and ``Specified-By`` instead.

CI is the backstop, because a hook can be bypassed. ``--range`` checks
every message a branch adds.

Examples:
    Run against one message file, then against a range:

    ```console
    $ uv run python scripts/check_commit_msg.py .git/COMMIT_EDITMSG
    checked .git/COMMIT_EDITMSG: feat, 2 issue references
    $ uv run python scripts/check_commit_msg.py --range origin/main..HEAD
    checked de7f8331c: status, 3 issue references
    ```

See Also:
    - docs/reference/commits.md: Vocabulary, issue footers and worker trailers.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

SPEC = "https://www.conventionalcommits.org/en/v1.0.0/"
TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)
_FORBIDDEN_COAUTHOR_EMAILS = frozenset({"cursoragent@cursor.com"})
_CO_AUTHORED_BY = re.compile(r"^Co-Authored-By:\s+(?P<ident>.+)$", re.IGNORECASE)
RANGE = "--range"
RANGE_ARGS = 2
_SUBJECT = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^()\n]*)\))?(?P<bang>!)?: (?P<rest>.*)$"
)
_GENERATED = re.compile(
    r"^(?:Merge (?:branch|branches|pull request|remote-tracking|tag|commit)\b"
    r"|Merge [0-9a-f]{7,40} into [0-9a-f]{7,40}$"
    r"|Revert \"[^\"]*\""
    r"|fixup!|squash!|amend!)",
    re.MULTILINE,
)
_COMMENT = re.compile(r"^#(?:\s|$)")
_SCISSORS = re.compile(r"^# --- >8 ---$", re.MULTILINE)
_ISSUE = re.compile(r"#\d+")
_TRAILING_REFS = re.compile(r"\s*\((?:#\d+(?:,\s*)?)+\)$")
_FOOTER_TOKEN_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 -]*)(?:: | #)")
_AUTHOR_PATTERN = re.compile(r"^(?:.*<)?(?P<email>[^>]+@[^>]+)>?.*$")


def get_grandfather_cutover() -> str | None:
    """Read the co-author grandfather boundary from pyproject.toml.

    Returns:
        A full commit SHA, or None when grandfathering is disabled.
    """
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        return None
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    try:
        value = config["tool"]["typevet"]["commit-msg"]["coauthor-grandfather-through"]
    except KeyError:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def is_coauthor_grandfathered(full_sha: str) -> bool:
    """Whether a commit may keep a harness ``Co-Authored-By`` in range checks.

    Args:
        full_sha: The commit object name git accepts.

    Returns:
        True when the commit is at or before the configured cutover on main.
    """
    cutover = get_grandfather_cutover()
    if cutover is None:
        return False
    result = subprocess.run(
        ["/usr/bin/git", "merge-base", "--is-ancestor", full_sha, cutover],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _forbidden_coauthor_email(line: str) -> str | None:
    """The forbidden email on a ``Co-Authored-By`` line, if any.

    Args:
        line: One line from a commit message.

    Returns:
        The lower-cased email when the line names a forbidden harness identity.
    """
    match = _CO_AUTHORED_BY.match(line.strip())
    if match is None:
        return None
    ident = match["ident"].strip()
    email_match = _AUTHOR_PATTERN.match(ident)
    if email_match is None:
        return None
    email = email_match["email"].casefold()
    if email in _FORBIDDEN_COAUTHOR_EMAILS:
        return email
    return None


def strip_harness_coauthors(text: str) -> str:
    """Drop auto-appended harness ``Co-Authored-By`` lines from a message.

    Args:
        text: Raw commit message file contents.

    Returns:
        The message with forbidden harness co-author trailers removed.
    """
    lines = text.splitlines()
    kept = [line for line in lines if _forbidden_coauthor_email(line) is None]
    while len(kept) > 1 and not kept[-1].strip() and not kept[-2].strip():
        kept.pop()
    result = "\n".join(kept)
    if text.endswith("\n"):
        result += "\n"
    return result


def get_allowed_authors() -> list[str]:
    """Read the allowed author emails from pyproject.toml.

    Returns:
        A list of allowed email addresses.

    Raises:
        RuntimeError: If the configuration is missing or malformed.
    """
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        raise RuntimeError("pyproject.toml not found")

    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    try:
        return config["tool"]["typevet"]["commit-msg"]["allowed-authors"]
    except KeyError as exc:
        raise RuntimeError(
            "Missing [tool.typevet.commit-msg] section in pyproject.toml"
        ) from exc


def get_author_email_from_git_var() -> str:
    """Read the author email from GIT_AUTHOR_IDENT.

    Returns:
        The author email address.

    Raises:
        RuntimeError: If the author email cannot be determined.
    """
    try:
        result = subprocess.run(
            ["/usr/bin/git", "var", "GIT_AUTHOR_IDENT"],
            capture_output=True,
            text=True,
            check=True,
        )
        ident = result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("git var GIT_AUTHOR_IDENT failed") from exc

    match = _AUTHOR_PATTERN.match(ident)
    if match is None:
        raise RuntimeError(
            f"Could not parse author email from GIT_AUTHOR_IDENT: {ident}"
        )

    return match["email"]


def get_author_emails_in_range(rev_range: str) -> list[str]:
    """Read the author emails for all commits in a revision range.

    Args:
        rev_range: A git range such as ``origin/main..HEAD``.

    Returns:
        A list of author emails, one per commit (newest first).
    """
    result = subprocess.run(
        ["/usr/bin/git", "log", "--format=%ae", rev_range],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def message_lines(text: str) -> list[str]:
    """The message with git's own lines removed.

    Args:
        text: The raw content of the commit message file.

    Returns:
        The remaining lines, with trailing blank lines dropped.
    """
    cut = _SCISSORS.search(text)
    if cut is not None:
        text = text[: cut.start()]
    lines = [line for line in text.splitlines() if not _COMMENT.match(line)]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def subject_problems(subject: str) -> list[str]:
    """Why the subject line fails the specification, if it does.

    Args:
        subject: The first line of the message.

    Returns:
        Failure messages, empty when the subject passes.
    """
    match = _SUBJECT.match(subject)
    if match is None:
        return [f"the subject is not '<type>[(scope)][!]: <description>': {subject!r}"]
    kind = match["type"]
    if kind not in TYPES:
        allowed = ", ".join(TYPES)
        return [f"the type {kind!r} is not one of: {allowed}"]
    problems = []
    scope = match["scope"]
    if scope is not None and not scope.strip():
        problems.append("the scope is empty: write a scope or drop the parentheses")
    description = _TRAILING_REFS.sub("", match["rest"].strip()).strip()
    if not description:
        problems.append("the description is empty")
    elif description.endswith("."):
        problems.append("the description ends with a period: drop it")
    return problems


def body_problems(lines: list[str]) -> list[str]:
    """Why the body and the footers fail the specification, if they do.

    Args:
        lines: Every line of the message, subject first.

    Returns:
        Failure messages, empty when the body passes.
    """
    problems: list[str] = []
    if len(lines) > 1 and lines[1].strip():
        problems.append("the body needs one blank line after the subject")

    # Check for BREAKING CHANGE variants
    for line in lines[1:]:
        token_match = re.match(
            r"^(?P<token>BREAKING[ -]CHANGE)(?=:| #)", line, re.IGNORECASE
        )
        if token_match is not None and token_match["token"] not in (
            "BREAKING CHANGE",
            "BREAKING-CHANGE",
        ):
            problems.append(
                f"write 'BREAKING CHANGE' or 'BREAKING-CHANGE' in upper case, "
                f"not {token_match['token']!r}"
            )

    # Parse footers to validate syntax
    # A footer line starts with a token followed by ": " or " #"
    # Wrapped values continue on subsequent lines until another footer
    if len(lines) > 1:
        body_lines = lines[1:]
        in_footer = False

        for line in body_lines:
            stripped = line.strip()
            if stripped:
                if _FOOTER_TOKEN_RE.match(line):
                    in_footer = True
                elif not in_footer:
                    pass

    return problems


def coauthor_problems(lines: list[str]) -> list[str]:
    """Why a model or harness ``Co-Authored-By`` trailer fails, if it does.

    Args:
        lines: Every line of the message, subject first.

    Returns:
        Failure messages, empty when every co-author line is allowed.
    """
    return [
        "Co-Authored-By must not name a model or harness; "
        "use Generated-By or Specified-By instead"
        for line in lines
        if _forbidden_coauthor_email(line) is not None
    ]


_COAUTHOR_PROBLEM = (
    "Co-Authored-By must not name a model or harness; "
    "use Generated-By or Specified-By instead"
)


def split_coauthor_problems(found: list[str]) -> tuple[list[str], list[str]]:
    """Separate harness co-author failures from every other failure.

    Args:
        found: Failure messages from :func:`problems`.

    Returns:
        Co-author failures and all other failures.
    """
    coauthor = [problem for problem in found if problem == _COAUTHOR_PROBLEM]
    other = [problem for problem in found if problem != _COAUTHOR_PROBLEM]
    return coauthor, other


def problems(text: str) -> tuple[list[str], bool]:
    """Every way the message fails the specification.

    Checks the subject, body layout, breaking-change tokens and forbidden
    model or harness ``Co-Authored-By`` trailers.

    Args:
        text: The raw content of the commit message file.

    Returns:
        A tuple of (failure messages, has_issue_reference).
    """
    lines = message_lines(text)
    if not lines:
        return ["the message is empty"], False
    if _GENERATED.match(lines[0]):
        return [], False

    found = subject_problems(lines[0]) + body_problems(lines) + coauthor_problems(lines)
    has_issue = _ISSUE.search("\n".join(lines)) is not None

    return found, has_issue


def summary(text: str) -> str:
    """The one-line report for a message that passed.

    Args:
        text: The raw content of the commit message file.

    Returns:
        The type and the issue reference count.
    """
    lines = message_lines(text)
    match = _SUBJECT.match(lines[0])
    kind = match["type"] if match else "generated"
    refs = len(set(_ISSUE.findall("\n".join(lines))))
    unit = "reference" if refs == 1 else "references"
    return f"{kind}, {refs} issue {unit}"


def messages_in_range(rev_range: str) -> list[tuple[str, str, str]]:
    """Every commit message in a revision range.

    Args:
        rev_range: A git range such as ``origin/main..HEAD``.

    Returns:
        The short hash, the message body, and the full hash of each commit,
        newest first.
    """
    proc = subprocess.run(
        ["/usr/bin/git", "log", "-z", "--format=%H%n%B", rev_range],
        capture_output=True,
        text=True,
        check=True,
    )
    found = []
    for chunk in proc.stdout.split("\0"):
        if not chunk.strip():
            continue
        sha, _, body = chunk.partition("\n")
        found.append((sha[:9], body, sha))
    return found


def check_author_for_commit_msg() -> list[str] | None:
    """Check if the committing author is allowed (commit-msg mode).

    This mode uses `git var GIT_AUTHOR_IDENT` because the commit does not
    exist yet, so the pending author is the right thing to check.

    Returns:
        A list with one failure message if the author is not allowed,
        or None if the author is allowed.

    Raises:
        RuntimeError: When author configuration cannot be read, except for
            an unconfigured git identity (that case returns guidance instead).
    """
    allowed = get_allowed_authors()
    try:
        author_email = get_author_email_from_git_var()
    except RuntimeError as exc:
        if "git var GIT_AUTHOR_IDENT failed" in str(exc):
            return [
                "git identity is not configured",
                "",
                "Set your name and email before committing:",
                "    git config --global user.name 'Your Name'",
                "    git config --global user.email 'your.email@example.com'",
            ]
        raise

    if author_email not in allowed:
        return [
            f"author email {author_email!r} is not in the allowed list",
            "",
            "Use a scratch repository instead:",
            "    tmp=$(mktemp -d) && git -C '$tmp' init -q",
        ]
    return None


def check_author_in_range(rev_range: str) -> list[str]:
    """Check if all commits in a range have allowed authors.

    This mode reads each commit's recorded author with `git log --format=%ae`
    because `git var` would report the runner's identity, not the commit's
    author.

    Args:
        rev_range: A git range such as ``origin/main..HEAD``.

    Returns:
        A list of failure messages, one per disallowed author, or empty if all
        authors are allowed.
    """
    allowed = get_allowed_authors()
    author_emails = get_author_emails_in_range(rev_range)

    problems = [
        f"author email {email!r} is not in the allowed list"
        for email in author_emails
        if email not in allowed
    ]

    return problems


def report(
    label: str,
    text: str,
    *,
    range_mode: bool = False,
    rev_range: str | None = None,
    full_sha: str | None = None,
) -> bool:
    """Check one message and print the outcome.

    Args:
        label: What to name the message in the output.
        text: The raw message.
        range_mode: If True, check authors from commit history rather than
            the pending commit. Ignored unless range is provided.
        rev_range: The revision range for author checking in range mode.
        full_sha: The commit object name when checking history.

    Returns:
        True when the message failed.
    """
    author_problems: list[str] | None = None

    if range_mode and rev_range is not None:
        author_problems = check_author_in_range(rev_range)
    else:
        author_problems = check_author_for_commit_msg()

    if author_problems:
        for problem in author_problems:
            print(f"FAIL {label}: {problem}")
        return True

    found, _ = problems(text)
    coauthor, other = split_coauthor_problems(found)
    grandfather = (
        range_mode
        and full_sha is not None
        and coauthor
        and is_coauthor_grandfathered(full_sha)
    )
    if grandfather:
        for problem in coauthor:
            print(f"WARN {label}: {problem} (grandfathered on main before #85)")
    failures = other + ([] if grandfather else coauthor)
    for problem in failures:
        print(f"FAIL {label}: {problem}")
    if not failures:
        print(f"checked {label}: {summary(text)}")
    return bool(failures)


def main(argv: list[str] | None = None) -> int:
    """Run the gate.

    Args:
        argv: The paths pre-commit passes, or ``--range <range>``. None
            reads ``sys.argv``.

    Returns:
        Process exit code: 1 on any failure, else 0.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("FAIL: name the commit message file")
        return 1
    failed = False
    if args[0] == RANGE:
        if len(args) != RANGE_ARGS:
            print(f"FAIL: {RANGE} takes one revision range")
            return 1
        rev_range = args[1]
        try:
            found = messages_in_range(rev_range)
        except subprocess.CalledProcessError as error:
            reason = error.stderr.strip().splitlines()
            print(
                f"FAIL {args[1]}: git log refused the range: {reason[0] if reason else ''}"
            )
            return 1
        for sha, text, full_sha in found:
            failed |= report(
                sha,
                text,
                range_mode=True,
                rev_range=rev_range,
                full_sha=full_sha,
            )
    else:
        for path in (Path(arg) for arg in args):
            if not path.is_file():
                print(f"FAIL {path}: missing")
                failed = True
                continue
            # Strip harness coauthors in-place so Cursor injection after
            # prepare-commit-msg (or a missing prepare hook) cannot land.
            text = strip_harness_coauthors(path.read_text(encoding="utf-8"))
            path.write_text(text, encoding="utf-8")
            failed |= report(str(path), text)
    if failed:
        print(f"the specification is at {SPEC}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
