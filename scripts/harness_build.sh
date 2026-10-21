#!/usr/bin/env bash
# Run one builder brief on an external harness: Cursor or Codex.
# Usage: scripts/harness_build.sh <cursor|codex> <worktree> <brief-file> [model] [effort]
# Procedure: .claude/skills/delegate-to-harness/SKILL.md.
# The brief must contain a line starting "Accepted contract:" and the line "Do not commit.".
# The script prints the receipt path, the agent's result, its usage, the status,
# the diff stat against HEAD and the untracked files.
# The worktree must be an isolated git worktree, not the checkout that runs the script.
# On Cursor the worktree must carry the tracked .cursor/cli.json deny list.
# The raw agent output goes to ${XDG_STATE_HOME:-$HOME/.local/state}/typevet/harness/.
# The supervisor accepts the gates and the diff, never the result text.
# A refusing pre-commit hook guards commits. The script warns when HEAD moved.
# The script stops a run after 1800 seconds and exits with the agent's exit code.
# Neither sandbox stops a write outside the worktree, for example to /tmp.
set -euo pipefail

usage() {
	echo "usage: $0 <cursor|codex> <worktree> <brief-file> [model] [effort]" >&2
	echo "cursor: model defaults to auto. Cursor ignores effort." >&2
	echo "codex: model defaults to ~/.codex/config.toml. Effort defaults to medium." >&2
	exit 2
}

[ "$#" -ge 3 ] && [ "$#" -le 5 ] || usage
harness="$1"
worktree="$2"
brief="$3"
model="${4:-}"
effort="${5:-medium}"
[ -d "$worktree" ] || usage
[ -f "$brief" ] || usage
case "$harness" in
cursor | codex) ;;
*) usage ;;
esac

# The hook guard below owns GIT_CONFIG_COUNT. Refuse to overwrite a caller's value.
# Check it before any git call, because a partial GIT_CONFIG_* set breaks git.
if [ -n "${GIT_CONFIG_COUNT:-}" ]; then
	echo "harness_build: FAIL: GIT_CONFIG_COUNT is already set. Unset it first." >&2
	exit 1
fi

# Refuse a directory outside git, and refuse a main checkout wherever the script starts.
# A linked worktree has its own git dir under the common dir. A main checkout has one dir for both.
if ! wt_git_dir="$(git -C "$worktree" rev-parse --absolute-git-dir 2>/dev/null)"; then
	echo "harness_build: FAIL: $worktree is not a git worktree." >&2
	exit 1
fi
wt_common_dir="$(cd "$worktree" && realpath "$(git rev-parse --git-common-dir)")"
if [ "$(realpath "$wt_git_dir")" = "$wt_common_dir" ]; then
	echo "harness_build: FAIL: refuse to run on the main checkout; use an isolated worktree." >&2
	exit 1
fi

# The Cursor CLI reads AGENTS.md and CLAUDE.md at the project root, and Codex reads AGENTS.md.
# The brief still carries the contract and the commit rule, because the brief is the contract.
if ! grep -q '^Accepted contract:' "$brief"; then
	echo "harness_build: FAIL: the brief lacks a line starting \"Accepted contract:\"." >&2
	exit 2
fi
if ! grep -qx 'Do not commit\.' "$brief"; then
	echo "harness_build: FAIL: the brief lacks the line \"Do not commit.\"." >&2
	exit 2
fi

# Cursor needs the tracked deny list, the project floor. The script never writes one.
if [ "$harness" = cursor ] &&
	! git -C "$worktree" ls-files --error-unmatch .cursor/cli.json >/dev/null 2>&1; then
	if [ -e "$worktree/.cursor/cli.json" ]; then
		echo "harness_build: FAIL: untracked .cursor/cli.json exists; remove it first." >&2
	else
		echo "harness_build: FAIL: the worktree has no tracked .cursor/cli.json; the project floor is required." >&2
	fi
	exit 1
fi

receipts="${XDG_STATE_HOME:-$HOME/.local/state}/typevet/harness"
mkdir -p "$receipts"
receipt="$receipts/$(date -u +%Y%m%dT%H%M%SZ)-$$-$harness.jsonl"
echo "receipt: $receipt"

last="$(mktemp)"
hooks="$(mktemp -d)"
trap 'rm -rf "$last" "$hooks"' EXIT

# Commit guard for both harnesses: a pre-commit hook that refuses every commit.
# A chained "cat x && git commit" can get past the Cursor deny file.
# The environment sets core.hooksPath. The repository config stays unchanged.
# "git commit --no-verify" skips the hook. The HEAD check after the run reports that case.
printf '#!/bin/sh\necho "harness_build: commits are blocked. The supervisor commits." >&2\nexit 1\n' \
	>"$hooks/pre-commit"
chmod +x "$hooks/pre-commit"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0="$hooks"

head_before="$(git -C "$worktree" rev-parse HEAD)"
status=0
case "$harness" in
cursor)
	timeout --kill-after=30 1800 agent -p --trust --force --sandbox enabled --output-format json \
		--model "${model:-auto}" --workspace "$worktree" "$(cat "$brief")" \
		>"$receipt" || status=$?
	echo "result:"
	jq -r '.result' "$receipt" 2>/dev/null || cat "$receipt"
	echo "usage:"
	jq -r 'if .usage == null then "unknown" else (.usage|tojson) end' "$receipt" 2>/dev/null || echo unknown
	;;
codex)
	# Effort levels: low, medium, high and xhigh. The xhigh string is unverified.
	model_args=()
	if [ -n "$model" ]; then model_args=(-m "$model"); fi
	timeout --kill-after=30 1800 codex exec -C "$worktree" -s workspace-write \
		-c approval_policy=never -c model_reasoning_effort="$effort" \
		"${model_args[@]}" --json -o "$last" - <"$brief" >"$receipt" || status=$?
	jq -Rs -c '{type: "harness_build.last_message", text: .}' "$last" >>"$receipt"
	echo "result:"
	cat "$last"
	echo
	echo "usage:"
	codex_usage="$(jq -c 'select(.type == "turn.completed") | .usage' "$receipt" 2>/dev/null || true)"
	echo "${codex_usage:-unknown}"
	jq -c 'select(.type == "turn.failed" or .type == "error")' "$receipt" 2>/dev/null || true
	;;
esac

if [ "$status" = 124 ]; then
	echo "harness timed out (exit 124)"
fi
# A moved HEAD means a commit got past the hook, for example with --no-verify.
head_after="$(git -C "$worktree" rev-parse HEAD)"
if [ "$head_after" != "$head_before" ]; then
	echo "WARNING: HEAD moved from $head_before to $head_after; a commit slipped past the guard" >&2
fi
echo "status:"
git -C "$worktree" status --short
echo "diff:"
git -C "$worktree" diff HEAD --stat
echo "untracked:"
git -C "$worktree" ls-files --others --exclude-standard
exit "$status"
