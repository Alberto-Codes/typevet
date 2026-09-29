r"""Fingerprints that distinguish evaluation smoke runs ([#186][i186]).

Captures repo baseline, working-tree drift, prompt bundles, pinned code and
fixture bytes, runtime labels and per-arm call counts. Every run gets a unique
``run_id``; the ``identity_digest`` excludes it so two prompt variants at the
same commit stay distinguishable. The baseline reader resolves ``HEAD`` in a
plain checkout or a linked worktree, from loose refs or ``packed-refs``
([#209][i209]), without a git subprocess.

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.experiment_identity import (
        ExperimentIdentityRequest,
        PromptSpec,
        RuntimeBuild,
        WorkingTreeState,
        capture_experiment_identity,
        capture_working_tree_at_run_start,
        finalize_experiment_identity,
        snapshot_evaluated_inputs,
        write_receipt_exclusive,
    )

    root = Path(".")
    tree = capture_working_tree_at_run_start(root, porcelain=" M dirty.py\n")
    identity = capture_experiment_identity(
        ExperimentIdentityRequest(
            repo_root=root,
            prompts=(PromptSpec("expense", ("a",), "instructions", {"a": "rule"}),),
            code_paths={},
            fixture_paths={},
            runtime=RuntimeBuild("model", "template", "unknown"),
            arm_call_counts={"combined": 1},
            working_tree=tree,
        )
    )
    assert identity.run_id
    ```

See Also:
    - [typevet.evaluation.datasets.cord_expense][]: CORD expense smoke prompts
    - evals/tests/live/test_cord_expense_smoke_live.py: live receipt wiring

[i186]: https://github.com/Alberto-Codes/typevet/issues/186
[i209]: https://github.com/Alberto-Codes/typevet/issues/209
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_PORCELAIN_PREFIX_LEN = 4
_COMMIT_HEX = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 hex digest of one file's bytes.

    Returns:
        Lowercase hex SHA-256 of the file contents.
    """
    return _sha256_bytes(path.read_bytes())


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One named prompt bundle whose wording pins experiment identity.

    Attributes:
        name (str): Stable key in receipts, for example ``expense``.
        label_order (tuple[str, ...]): Label order the judge sees.
        instructions (str): Instruction text.
        criteria (Mapping[str, str]): Criterion text per label or answer key.


    Examples:
        ```python
        PromptSpec("expense", ("match",), "instructions", {"match": "rule"})
        ```
    """

    name: str
    label_order: tuple[str, ...]
    instructions: str
    criteria: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class WorkingTreeState:
    """Git baseline and optional dirty working tree at run start.

    Attributes:
        baseline_commit (str): ``HEAD`` commit at capture time.
        dirty (bool): Whether the working tree had unstaged or staged edits.
        dirty_paths (tuple[str, ...]): Repo-relative paths that differed.
        dirty_digest (str): Digest of dirty file bytes; empty when clean.


    Examples:
        ```python
        WorkingTreeState("0" * 40, False, (), "")
        ```
    """

    baseline_commit: str
    dirty: bool
    dirty_paths: tuple[str, ...]
    dirty_digest: str


@dataclass(frozen=True, slots=True)
class RuntimeBuild:
    """Model and router labels recorded on the receipt.

    Attributes:
        model (str): Model id the smoke called.
        served_template (str): Template family the adapter used.
        server_build (str): Router build string, or ``unknown`` when missing.
        template_identity (str): Exact template or jinja pin, or ``unknown``.
        projector_identity (str): Vision projector path or digest, or ``unknown``.
        config_identity (str): Server preset or config pin, or ``unknown``.


    Examples:
        ```python
        RuntimeBuild("model", "native_gemma3_turn", "unknown")
        ```
    """

    model: str
    served_template: str
    server_build: str
    template_identity: str = "unknown"
    projector_identity: str = "unknown"
    config_identity: str = "unknown"


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    """Full identity for one smoke run.

    Attributes:
        run_id (str): Unique id for this run attempt.
        baseline_commit (str): Git commit at capture time.
        working_tree (WorkingTreeState): Dirty-tree fingerprint.
        prompt_digests (Mapping[str, str]): Digest per ``PromptSpec.name``.
        code_path_digests (Mapping[str, str]): Digest per named source path.
        fixture_digests (Mapping[str, str]): Digest per named fixture path.
        runtime (RuntimeBuild): Model and template labels.
        arm_call_counts (Mapping[str, int]): Scoring calls per modality or arm.


    Examples:
        ```python
        tree = WorkingTreeState("0" * 40, False, (), "")
        ExperimentIdentity(
            run_id="r1",
            baseline_commit="0" * 40,
            working_tree=tree,
            prompt_digests={},
            code_path_digests={},
            fixture_digests={},
            runtime=RuntimeBuild("m", "t", "unknown"),
            arm_call_counts={"combined": 0},
        )
        ```
    """

    run_id: str
    baseline_commit: str
    working_tree: WorkingTreeState
    prompt_digests: Mapping[str, str]
    code_path_digests: Mapping[str, str]
    fixture_digests: Mapping[str, str]
    runtime: RuntimeBuild
    arm_call_counts: Mapping[str, int]

    def to_receipt_mapping(self) -> dict[str, object]:
        """Serialize this identity for a live smoke receipt JSON object.

        Returns:
            JSON-ready mapping with ``identity_digest``, working-tree dirty
            flags, and runtime fields including ``template_identity``,
            ``projector_identity`` and ``config_identity`` (``unknown`` when
            not resolved).
        """
        return {
            "run_id": self.run_id,
            "baseline_commit": self.baseline_commit,
            "working_tree": {
                "dirty": self.working_tree.dirty,
                "dirty_paths": list(self.working_tree.dirty_paths),
                "dirty_digest": self.working_tree.dirty_digest,
            },
            "prompt_digests": dict(self.prompt_digests),
            "code_path_digests": dict(self.code_path_digests),
            "fixture_digests": dict(self.fixture_digests),
            "runtime": {
                "model": self.runtime.model,
                "served_template": self.runtime.served_template,
                "server_build": self.runtime.server_build,
                "template_identity": self.runtime.template_identity,
                "projector_identity": self.runtime.projector_identity,
                "config_identity": self.runtime.config_identity,
            },
            "arm_call_counts": dict(self.arm_call_counts),
            "identity_digest": identity_digest(self),
        }


def prompt_digest(spec: PromptSpec) -> str:
    """Hash one prompt bundle so label order and wording pin identity.

    Returns:
        Lowercase hex SHA-256 of the canonical prompt JSON payload.
    """
    payload = {
        "name": spec.name,
        "label_order": list(spec.label_order),
        "instructions": spec.instructions,
        "criteria": {key: spec.criteria[key] for key in sorted(spec.criteria)},
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))


def identity_digest(identity: ExperimentIdentity) -> str:
    """Return a stable digest that excludes ``run_id``.

    Returns:
        Lowercase hex SHA-256 of prompts, paths, baseline, working tree,
        arm call counts and runtime labels including unknown identity slots.
    """
    payload = {
        "arm_call_counts": {
            key: identity.arm_call_counts[key]
            for key in sorted(identity.arm_call_counts)
        },
        "baseline_commit": identity.baseline_commit,
        "code_path_digests": {
            key: identity.code_path_digests[key]
            for key in sorted(identity.code_path_digests)
        },
        "fixture_digests": {
            key: identity.fixture_digests[key]
            for key in sorted(identity.fixture_digests)
        },
        "prompt_digests": {
            key: identity.prompt_digests[key] for key in sorted(identity.prompt_digests)
        },
        "runtime": {
            "model": identity.runtime.model,
            "served_template": identity.runtime.served_template,
            "server_build": identity.runtime.server_build,
            "template_identity": identity.runtime.template_identity,
            "projector_identity": identity.runtime.projector_identity,
            "config_identity": identity.runtime.config_identity,
        },
        "working_tree": {
            "dirty": identity.working_tree.dirty,
            "dirty_digest": identity.working_tree.dirty_digest,
            "dirty_paths": list(identity.working_tree.dirty_paths),
        },
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))


@dataclass(frozen=True, slots=True)
class ExperimentIdentityRequest:
    """Inputs that pin one smoke run besides optional overrides.

    Attributes:
        repo_root (Path): Repository root for resolving relative paths.
        prompts (Sequence[PromptSpec]): Named prompt bundles for the run.
        code_paths (Mapping[str, Path]): Named production files to digest.
        fixture_paths (Mapping[str, Path]): Named fixture files to digest.
        runtime (RuntimeBuild): Model and template labels.
        arm_call_counts (Mapping[str, int]): Scoring calls per arm name.
        working_tree (WorkingTreeState): Baseline commit and dirty fingerprint
            captured by the caller at run start (no git subprocess here).


    Examples:
        ```python
        tree = WorkingTreeState("0" * 40, False, (), "")
        ExperimentIdentityRequest(
            repo_root=Path("."),
            prompts=(),
            code_paths={},
            fixture_paths={},
            runtime=RuntimeBuild("m", "t", "unknown"),
            arm_call_counts={},
            working_tree=tree,
        )
        ```
    """

    repo_root: Path
    prompts: Sequence[PromptSpec]
    code_paths: Mapping[str, Path]
    fixture_paths: Mapping[str, Path]
    runtime: RuntimeBuild
    arm_call_counts: Mapping[str, int]
    working_tree: WorkingTreeState


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _resolve_gitdir(repo_root: Path) -> Path | None:
    dot_git = repo_root / ".git"
    if dot_git.is_dir():
        return dot_git
    text = _read(dot_git)
    if text is None or not text.startswith("gitdir:"):
        return None
    return repo_root / text.removeprefix("gitdir:").strip()


def _ref_search_dirs(gitdir: Path) -> list[Path] | None:
    commondir = gitdir / "commondir"
    if not commondir.exists():
        return [gitdir]
    common = _read(commondir)
    return None if common is None else [gitdir, gitdir / common]


def _packed_ref(git_dir: Path, ref: str) -> str | None:
    packed = git_dir / "packed-refs"
    text = _read(packed) if packed.exists() else ""
    if text is None:
        return "unknown"
    for line in text.splitlines():
        sha, _, name = line.partition(" ")
        if not line.startswith(("#", "^")) and name.strip() == ref:
            return sha.strip()
    return None


def _resolve_head(gitdir: Path) -> str | None:
    head = _read(gitdir / "HEAD")
    if head is None or not head.startswith("ref:"):
        return head
    ref = head.removeprefix("ref:").strip()
    if not ref.startswith("refs/") or ".." in ref.split("/"):
        return None
    for git_dir in _ref_search_dirs(gitdir) or []:
        loose = git_dir / ref
        if loose.exists():
            return _read(loose)
        packed = _packed_ref(git_dir, ref)
        if packed is not None:
            return packed
    return None


def read_baseline_commit(repo_root: Path) -> str:
    """Return ``HEAD`` commit hex from git metadata without a subprocess.

    Handles a ``.git`` directory or a linked-worktree ``.git`` file
    (``gitdir: <path>``). A branch ref under ``refs/`` resolves as a loose ref
    or from ``packed-refs``, in the gitdir first and then in the ``commondir``.

    Returns:
        Lowercase 40- or 64-hex commit, or ``unknown`` when ``HEAD`` or its
        ref cannot be read or resolved, or the value is not a commit hex.
    """
    gitdir = _resolve_gitdir(repo_root)
    commit = None if gitdir is None else _resolve_head(gitdir)
    if commit is None or not _COMMIT_HEX.fullmatch(commit):
        return "unknown"
    return commit


def _porcelain_path(line: str) -> str | None:
    if len(line) < _PORCELAIN_PREFIX_LEN:
        return None
    raw = line[3:].strip()
    if not raw:
        return None
    return raw.split(" -> ", 1)[-1].strip()


def capture_working_tree_at_run_start(
    repo_root: Path,
    *,
    porcelain: str,
) -> WorkingTreeState:
    """Fingerprint ``HEAD`` and working-tree drift before inference.

    The caller supplies ``git status --porcelain`` text (live harness or tests).
    This module does not spawn git.

    Args:
        repo_root: Repository root for the smoke harness checkout.
        porcelain: Output of ``git status --porcelain`` captured by the caller.

    Returns:
        Baseline commit and dirty paths derived from ``porcelain``.
    """
    baseline = read_baseline_commit(repo_root)
    if baseline == "unknown":
        return WorkingTreeState("unknown", True, (".git/HEAD",), "unknown")
    return working_tree_from_porcelain(
        baseline_commit=baseline,
        porcelain=porcelain,
        repo_root=repo_root,
    )


def working_tree_from_porcelain(
    *,
    baseline_commit: str,
    porcelain: str,
    repo_root: Path,
) -> WorkingTreeState:
    """Build a working-tree fingerprint from ``git status --porcelain`` text.

    Args:
        baseline_commit: Commit at capture time.
        porcelain: Output of ``git status --porcelain`` (may be empty).
        repo_root: Repository root (paths in ``porcelain`` are repo-relative).

    Returns:
        Clean state when ``porcelain`` is empty; otherwise dirty with paths and
        a digest of the porcelain lines.
    """
    _ = repo_root
    lines = [line for line in porcelain.splitlines() if line.strip()]
    if not lines:
        return WorkingTreeState(baseline_commit, False, (), "")
    paths = tuple(
        sorted({path for line in lines if (path := _porcelain_path(line)) is not None})
    )
    digest = _sha256_bytes("\n".join(sorted(lines)).encode("utf-8", "surrogateescape"))
    return WorkingTreeState(baseline_commit, True, paths, digest)


class ReceiptAlreadyExistsError(OSError):
    """Raised when a receipt path already exists and must stay immutable.

    Attributes:
        path (Path): Receipt file that already exists on disk.

    Examples:
        ```python
        raise ReceiptAlreadyExistsError(Path("/tmp/receipt-x.json"))
        ```
    """

    path: Path

    def __init__(self, path: Path) -> None:
        """Record the existing receipt path on the exception."""
        super().__init__(f"receipt already exists: {path}")
        self.path = path


@dataclass(frozen=True, slots=True)
class EvaluatedInputsSnapshot:
    """Prompt and evaluated file digests frozen before any scoring call.

    Attributes:
        prompt_digests (Mapping[str, str]): Digest per ``PromptSpec.name``.
        code_path_digests (Mapping[str, str]): Digest per named source path.
        fixture_digests (Mapping[str, str]): Digest per named fixture path,
            including receipt PNG bytes for multimodal smokes.

    Examples:
        ```python
        EvaluatedInputsSnapshot(
            {},
            {"cord_expense": "abc" * 21},
            {"receipt_R01": "def" * 21},
        )
        ```
    """

    prompt_digests: Mapping[str, str]
    code_path_digests: Mapping[str, str]
    fixture_digests: Mapping[str, str]


def snapshot_evaluated_inputs(
    *,
    prompts: Sequence[PromptSpec],
    code_paths: Mapping[str, Path],
    fixture_paths: Mapping[str, Path],
) -> EvaluatedInputsSnapshot:
    """Read prompt bundles and evaluated paths once, before inference.

    Finalization must use this snapshot instead of re-reading the same paths
    after scoring, so late edits cannot change recorded digests.

    Args:
        prompts: Named prompt bundles for the run.
        code_paths: Production files whose bytes pin identity.
        fixture_paths: Fixture files whose bytes pin identity.

    Returns:
        Frozen digests for finalize_experiment_identity.
    """
    return EvaluatedInputsSnapshot(
        prompt_digests={spec.name: prompt_digest(spec) for spec in prompts},
        code_path_digests={
            name: file_digest(path) for name, path in sorted(code_paths.items())
        },
        fixture_digests={
            name: file_digest(path) for name, path in sorted(fixture_paths.items())
        },
    )


def finalize_experiment_identity(
    *,
    run_start: RunIdentityStart,
    evaluated: EvaluatedInputsSnapshot,
    arm_call_counts: Mapping[str, int],
) -> ExperimentIdentity:
    """Build receipt identity from a pre-scoring snapshot and call counts.

    Args:
        run_start: Run id, working tree and runtime captured before scoring.
        evaluated: Digests from snapshot_evaluated_inputs; paths are not reread.
        arm_call_counts: Per-arm scoring calls recorded after the run.

    Returns:
        Complete identity for receipt serialization.
    """
    tree = run_start.working_tree
    return ExperimentIdentity(
        run_id=run_start.run_id,
        baseline_commit=tree.baseline_commit,
        working_tree=tree,
        prompt_digests=dict(evaluated.prompt_digests),
        code_path_digests=dict(evaluated.code_path_digests),
        fixture_digests=dict(evaluated.fixture_digests),
        runtime=run_start.runtime,
        arm_call_counts=dict(arm_call_counts),
    )


def write_receipt_exclusive(path: Path, receipt: Mapping[str, object]) -> None:
    """Write one receipt JSON file; fail when the path already exists.

    Args:
        path: Destination file (parent directories are created).
        receipt: JSON-serializable receipt body.

    Raises:
        ReceiptAlreadyExistsError: When ``path`` is already present.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ReceiptAlreadyExistsError(path) from exc


def cord_expense_receipt_path(output_dir: Path, attempt_id: str) -> Path:
    """Return the immutable receipt path for one CORD smoke attempt.

    Args:
        output_dir: Gitignored output directory for CORD receipts.
        attempt_id: Unique run or attempt id from experiment identity.

    Returns:
        Path ``receipt-<attempt_id>.json`` under ``output_dir``.
    """
    return output_dir / f"receipt-{attempt_id}.json"


@dataclass(frozen=True, slots=True)
class RunIdentityStart:
    """Identity captured before any scoring calls in a live smoke.

    Attributes:
        run_id (str): Unique id for this run attempt.
        working_tree (WorkingTreeState): Tree fingerprint at run start.
        runtime (RuntimeBuild): Model and template labels known before scoring.

    Examples:
        ```python
        tree = WorkingTreeState("0" * 40, False, (), "")
        begin_run_identity(
            repo_root=Path("."),
            runtime=RuntimeBuild("m", "native_gemma3_turn", "unknown"),
            working_tree=tree,
            run_id="attempt-1",
        )
        ```
    """

    run_id: str
    working_tree: WorkingTreeState
    runtime: RuntimeBuild


def begin_run_identity(
    *,
    repo_root: Path,
    runtime: RuntimeBuild,
    working_tree: WorkingTreeState,
    run_id: str | None = None,
) -> RunIdentityStart:
    """Capture run identity before inference begins.

    Args:
        repo_root: Repository root (reserved for future path checks).
        runtime: Model and router labels already resolved for the run.
        working_tree: Working-tree fingerprint captured before scoring.
        run_id: Optional fixed id; a new uuid is used when omitted.

    Returns:
        Snapshot to pair with post-run call counts and digests.
    """
    _ = repo_root
    return RunIdentityStart(
        run_id=run_id or uuid.uuid4().hex,
        working_tree=working_tree,
        runtime=runtime,
    )


def capture_experiment_identity(
    request: ExperimentIdentityRequest,
    *,
    run_id: str | None = None,
) -> ExperimentIdentity:
    """Capture run identity from an injected working-tree fingerprint.

    Reads evaluated paths once via ``snapshot_evaluated_inputs``, then
    ``finalize_experiment_identity`` with the supplied call counts.

    Args:
        request: Repo, prompts, paths, runtime, call counts and working tree.
        run_id: Optional fixed id; a new uuid is used when omitted.

    Returns:
        A complete ``ExperimentIdentity`` for receipt serialization.
    """
    evaluated = snapshot_evaluated_inputs(
        prompts=request.prompts,
        code_paths=request.code_paths,
        fixture_paths=request.fixture_paths,
    )
    start = RunIdentityStart(
        run_id=run_id or uuid.uuid4().hex,
        working_tree=request.working_tree,
        runtime=request.runtime,
    )
    return finalize_experiment_identity(
        run_start=start,
        evaluated=evaluated,
        arm_call_counts=request.arm_call_counts,
    )
