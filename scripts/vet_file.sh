#!/usr/bin/env bash
# PostToolUse hook: vet every Python file a tool call just changed.
# Runs ruff format, ruff check and docvet on each file, plus check_loc on
# files under src when that script exists. ty, lint-imports and pytest stay
# in the pre-commit hook and the CLAUDE.md gate table (#7).
# Reads the hook JSON on stdin. Write and Edit name the file. A Bash
# command reports the files it changed in tool_response.bashEditDiff
# when bashEditDiffEnabled is on. Without that list the hook takes every
# Python file under src, tests or scripts whose mtime falls inside the
# command's duration. Findings return as additionalContext. A clean file
# returns nothing, so a clean edit costs no tokens. Each gate's output is
# capped at 30 lines. Paths are canonical (Python realpath) and prefixed
# with ./ . Until pyproject.toml exists (uv is always the runner), the hook exits 0 with
# no output so early sessions are not blocked.
set -u
input=$(cat)
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null) || exit 0
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
canon() { python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
root=$(canon "$root") || exit 0
cd "$root" || exit 0

# No package yet: stay silent (typevet scaffolding phase).
[ -f pyproject.toml ] || exit 0
command -v uv >/dev/null 2>&1 || exit 0

# Print the project-relative ./path of one Python file inside the root.
inside() {
  local f="$1"
  case "$f" in *.py) ;; *) return 1 ;; esac
  case "$f" in /*) ;; *) f="$root/$f" ;; esac
  f=$(canon "$f") || return 1
  case "$f" in "$root"/*) f="./${f#"$root"/}" ;; *) return 1 ;; esac
  [ -f "$f" ] && printf '%s\n' "$f"
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
      ms=$(printf '%s' "$input" | jq -r '.duration_ms // 0' 2>/dev/null)
      candidates=$(python3 - "$ms" <<'PY'
import os, sys, time
try:
    window = float(sys.argv[1]) / 1000
except ValueError:
    window = 0.0
since = time.time() - window - 2
for top in ("src", "tests", "scripts"):
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
files=""
while IFS= read -r c; do
  [ -n "$c" ] || continue
  rel=$(inside "$c") && files+="$rel"$'\n'
done <<< "$candidates"
files=$(printf '%s' "$files" | sort -u)
[ -n "$files" ] || exit 0

have_docvet=0
uv run docvet --help >/dev/null 2>&1 && have_docvet=1
have_loc=0
[ -f scripts/check_loc.py ] && have_loc=1

# Print the gate findings for one file, or nothing when it is clean.
vet() {
  local rel="$1" one="" fmt chk doc loc
  fmt=$(uv run ruff format --check --diff "$rel" 2>&1) \
    || one+="ruff format:"$'\n'"$(printf '%s\n' "$fmt" | head -30)"$'\n'
  chk=$(uv run ruff check --no-fix --output-format concise "$rel" 2>&1) \
    || one+="ruff check:"$'\n'"$(printf '%s\n' "$chk" | head -30)"$'\n'
  if [ "$have_docvet" -eq 1 ]; then
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
out=""
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  one=$(vet "$rel")
  [ -n "$one" ] && out+="== $rel"$'\n'"$one"$'\n'
done <<< "$files"
[ -z "$out" ] && exit 0
jq -n --arg c "$out" '{hookSpecificOutput: {hookEventName: "PostToolUse",
  additionalContext: ("Gate findings. Fix them before the next gate run.\n" + $c)}}'
