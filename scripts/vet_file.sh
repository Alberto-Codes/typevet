#!/usr/bin/env bash
# PostToolUse hook: vet every Python file a tool call just changed.
# Runs ruff format, ruff check and docvet on each file, plus check_loc on
# files under src when that script exists. ty, lint-imports and pytest stay
# in the pre-commit hook and the CLAUDE.md gate table (#7).
# Reads the hook JSON on stdin. Write and Edit name the file. A Bash
# command reports the files it changed in tool_response.bashEditDiff
# when bashEditDiffEnabled is on. Without that list the hook takes every
# Python file under src, tests or scripts of the checkout that holds the
# hook cwd whose mtime falls inside the command's duration.
# Each file is vetted from the git toplevel of its own checkout, so an edit
# in a worktree under .claude/worktrees uses that worktree's config and
# paths. Only the project checkout and its git worktrees are vetted.
# docvet skips a file that the docvet exclude list in pyproject.toml names,
# as `docvet check --all` does (#228). Findings return as
# additionalContext. A clean file returns nothing, so a clean edit costs no
# tokens. Each gate's output is capped at 30 lines. A file in the project
# checkout shows as ./path; a file in another worktree shows its canonical
# absolute path. Until pyproject.toml exists (uv is always the runner), the
# hook exits 0 with no output so early sessions are not blocked.
set -u
# A git hook (such as pre-commit) exports GIT_DIR, GIT_INDEX_FILE and more.
# Clear every inherited GIT_* variable so each git call finds the checkout
# from the path it names, not from the caller's repository.
while IFS= read -r var; do
  case "$var" in GIT_*) unset "$var" ;; esac
done < <(compgen -e)
input=$(cat)
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null) || exit 0
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
canon() { python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
root=$(canon "$root") || exit 0
command -v uv >/dev/null 2>&1 || exit 0

# Print the canonical git common dir of the checkout that holds directory $1.
common_dir() {
  local c
  c=$(git -C "$1" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || return 1
  canon "$c"
}
root_common=$(common_dir "$root") || root_common=""

# Print the checkout toplevel for directory $1: the project root, or a git
# worktree that shares the project's repository. Fail for anything else.
checkout_of() {
  local d="$1" top
  top=$(git -C "$d" rev-parse --show-toplevel 2>/dev/null) || top=""
  if [ -n "$top" ]; then
    top=$(canon "$top") || return 1
    [ "$top" = "$root" ] && { printf '%s\n' "$root"; return 0; }
    [ -n "$root_common" ] && [ "$(common_dir "$d")" = "$root_common" ] \
      && { printf '%s\n' "$top"; return 0; }
    return 1
  fi
  d=$(canon "$d") || return 1
  case "$d/" in "$root"/*) printf '%s\n' "$root" ;; *) return 1 ;; esac
}

base=$(printf '%s' "$input" | jq -r '.cwd // empty' 2>/dev/null)
[ -n "$base" ] && [ -d "$base" ] || base="$root"

# Print "<toplevel><TAB>./<relative path>" for one Python file in a checkout.
locate() {
  local f="$1" top
  case "$f" in *.py) ;; *) return 1 ;; esac
  case "$f" in /*) ;; *) f="$base/$f" ;; esac
  f=$(canon "$f") || return 1
  [ -f "$f" ] || return 1
  top=$(checkout_of "$(dirname "$f")") || return 1
  case "$f" in "$top"/*) printf '%s\t./%s\n' "$top" "${f#"$top"/}" ;; *) return 1 ;; esac
}

candidates=""
case "$tool" in
  Write|Edit)
    candidates=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
    ;;
  Bash)
    candidates=$(printf '%s' "$input" \
      | jq -r '.tool_response.bashEditDiff.files[]?.filePath // empty' 2>/dev/null)
    if [ -z "$candidates" ]; then
      scan=$(checkout_of "$base") || scan="$root"
      ms=$(printf '%s' "$input" | jq -r '.duration_ms // 0' 2>/dev/null)
      candidates=$(python3 - "$ms" "$scan" <<'PY'
import os, sys, time
try:
    window = float(sys.argv[1]) / 1000
except ValueError:
    window = 0.0
since = time.time() - window - 2
for top in ("src", "tests", "scripts"):
    top = os.path.join(sys.argv[2], top)
    if not os.path.isdir(top):
        continue
    for base, _dirs, names in os.walk(top):
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            try:
                recent = os.path.getmtime(path) >= since
            except OSError:
                continue
            if recent:
                print(path)
PY
)
    fi
    ;;
  *) exit 0 ;;
esac
located=""
while IFS= read -r c; do
  [ -n "$c" ] || continue
  one=$(locate "$c") && located+="$one"$'\n'
done <<< "$candidates"
located=$(printf '%s' "$located" | sort -u)
[ -n "$located" ] || exit 0

# Read ./paths on stdin; print those that the docvet exclude list in
# ./pyproject.toml does not name. docvet's own loader and matcher apply
# the list, so the result agrees with `docvet check --all`.
docvet_included() {
  uv run python -c '
import sys
from pathlib import Path

from docvet.config import load_config
from docvet.discovery import _is_excluded

exclude = load_config(Path("pyproject.toml").resolve()).exclude
for rel in sys.stdin.read().splitlines():
    if rel and not _is_excluded(rel.removeprefix("./"), exclude):
        print(rel)
'
}

# Print the gate findings for one file, or nothing when it is clean.
vet() {
  local rel="$1" docvet_on="$2" one="" fmt chk doc loc
  fmt=$(uv run ruff format --check --diff "$rel" 2>&1) \
    || one+="ruff format:"$'\n'"$(printf '%s\n' "$fmt" | head -30)"$'\n'
  chk=$(uv run ruff check --no-fix --output-format concise "$rel" 2>&1) \
    || one+="ruff check:"$'\n'"$(printf '%s\n' "$chk" | head -30)"$'\n'
  if [ "$docvet_on" -eq 1 ]; then
    doc=$(uv run docvet check --quiet "$rel" 2>&1 | head -30)
    [ -n "$doc" ] && one+="docvet:"$'\n'"$doc"$'\n'
  fi
  if [ "$have_loc" -eq 1 ]; then
    case "$rel" in ./src/*)
      loc=$(uv run python scripts/check_loc.py "$(dirname "$rel")" 2>&1 | grep -F -- "${rel#./}" || true)
      [ -n "$loc" ] && one+="check_loc:"$'\n'"$loc"$'\n'
      ;;
    esac
  fi
  printf '%s' "$one"
}

# Vet the files of one checkout from its toplevel; print labelled findings.
vet_checkout() {
  local top="$1" rels="$2" note="" included="" rel label one out=""
  cd "$top" || return 0
  # No package yet: stay silent (typevet scaffolding phase).
  [ -f pyproject.toml ] || return 0
  local have_docvet=0
  uv run docvet --help >/dev/null 2>&1 && have_docvet=1
  have_loc=0
  [ -f scripts/check_loc.py ] && have_loc=1
  if [ "$have_docvet" -eq 1 ]; then
    if ! included=$(printf '%s\n' "$rels" | docvet_included 2>&1); then
      note="docvet exclude lookup failed, so docvet ran on every file:"$'\n'
      note+="$(printf '%s\n' "$included" | tail -5)"$'\n'
      included="$rels"
    fi
  fi
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    docvet_on=0
    printf '%s\n' "$included" | grep -Fqx -- "$rel" && docvet_on=1
    one=$(vet "$rel" "$docvet_on")
    [ -n "$one" ] || continue
    label="$rel"
    [ "$top" = "$root" ] || label="$top/${rel#./}"
    out+="== $label"$'\n'"$one"$'\n'
  done <<< "$rels"
  [ -n "$out" ] && printf '%s%s' "$note" "$out"
  return 0
}

out=""
tops=$(printf '%s\n' "$located" | cut -f1 | sort -u)
while IFS= read -r top; do
  [ -n "$top" ] || continue
  rels=$(printf '%s\n' "$located" | awk -F'\t' -v t="$top" '$1 == t { print $2 }')
  out+=$(vet_checkout "$top" "$rels")
  [ -n "$out" ] && out+=$'\n'
done <<< "$tops"
[ -n "$out" ] || exit 0
jq -n --arg c "$out" '{hookSpecificOutput: {hookEventName: "PostToolUse",
  additionalContext: ("Gate findings. Fix them before the next gate run.\n" + $c)}}'
