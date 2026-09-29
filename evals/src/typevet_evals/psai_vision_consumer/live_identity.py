"""Pre-dispatch experiment identity for consumer live proofs ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.psai_vision_consumer.live_identity import (
        consumer_fixture_paths,
        finalize_consumer_live_identity,
        start_consumer_live_identity,
    )

    paths = consumer_fixture_paths(
        Path("tests/fixtures/psai/vision_smoke"),
        fixture=...,
    )
    assert "manifest" in paths
    ```

See Also:
    - [typevet_evals.experiment_identity][]: snapshot helpers

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from typevet.evaluation.datasets.psai_vision import (
    VisionSmokeFixture,
    vision_smoke_manifest_path,
)
from typevet_evals.experiment_identity import (
    EvaluatedInputsSnapshot,
    RunIdentityStart,
    RuntimeBuild,
    begin_run_identity,
    capture_working_tree_at_run_start,
    finalize_experiment_identity,
    snapshot_evaluated_inputs,
)
from typevet_evals.psai_vision_consumer.accounting import FROZEN_CONSUMER_CASE_UIDS
from typevet_evals.psai_vision_consumer.dispatch import ConsumerDispatchLedger

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _git_porcelain() -> str:
    proc = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return proc.stdout if proc.returncode == 0 else ""


def consumer_fixture_paths(
    fixture_root: Path,
    fixture: VisionSmokeFixture,
) -> dict[str, Path]:
    """Return manifest and frozen image paths for identity snapshot.

    Args:
        fixture_root: Committed vision smoke directory.
        fixture: Loaded fixture set.

    Returns:
        Named paths for ``snapshot_evaluated_inputs``.
    """
    paths: dict[str, Path] = {"manifest": vision_smoke_manifest_path(fixture_root)}
    for example in fixture.examples:
        if example.unique_data_id in FROZEN_CONSUMER_CASE_UIDS:
            paths[f"image:{example.unique_data_id}"] = (
                fixture_root / example.screenshot.file_name
            )
    return paths


def _server_build(health: Mapping[str, Any]) -> str:
    for key in ("build", "version", "status"):
        value = health.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def start_consumer_live_identity(
    *,
    fixture_root: Path,
    fixture: VisionSmokeFixture,
    model: str,
    served_template: str,
    health: Mapping[str, Any],
) -> tuple[RunIdentityStart, EvaluatedInputsSnapshot]:
    """Capture run identity immediately before matrix dispatch.

    Args:
        fixture_root: Committed vision smoke directory.
        fixture: Loaded fixture set.
        model: Model id under test.
        served_template: Classified template family name.
        health: Router ``/health`` JSON body.

    Returns:
        Run start snapshot and evaluated-input digests (not yet finalized).
    """
    evaluated = snapshot_evaluated_inputs(
        prompts=(),
        code_paths={},
        fixture_paths=consumer_fixture_paths(fixture_root, fixture),
    )
    run_start = begin_run_identity(
        repo_root=_REPO_ROOT,
        runtime=RuntimeBuild(
            model=model,
            served_template=served_template,
            server_build=_server_build(health),
        ),
        working_tree=capture_working_tree_at_run_start(
            _REPO_ROOT,
            porcelain=_git_porcelain(),
        ),
    )
    return run_start, evaluated


def finalize_consumer_live_identity(
    *,
    run_start: RunIdentityStart,
    evaluated: EvaluatedInputsSnapshot,
    ledger: ConsumerDispatchLedger,
) -> dict[str, object]:
    """Finalize receipt identity using pre-dispatch digests and call counts.

    Args:
        run_start: Identity captured before matrix dispatch.
        evaluated: Fixture digests from the same pre-dispatch snapshot.
        ledger: Dispatch ledger after the matrix leg.

    Returns:
        JSON-ready identity mapping for receipt serialization.
    """
    return finalize_experiment_identity(
        run_start=run_start,
        evaluated=evaluated,
        arm_call_counts={
            "judgment": ledger.judgment_calls,
            "scoring": ledger.scoring_requests,
        },
    ).to_receipt_mapping()
