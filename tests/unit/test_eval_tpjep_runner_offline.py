"""Offline eight-task runner smoke (#106)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.judgment_contract import ContractJudgmentFake
from typevet.evaluation.tpjep.loader import EIGHT_TASK_IDS, load_eight_task_fixture
from typevet.evaluation.tpjep.runner import TpjepRunConfig, run_tpjep_with_receipt

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "tpjep"
    / "eight_task_smoke.jsonl"
).read_text(encoding="utf-8")


@pytest.mark.unit
def test_offline_eight_task_receipt_metadata_and_schedule() -> None:
    tasks = load_eight_task_fixture(_FIXTURE)
    fake = ContractJudgmentFake()
    receipt = run_tpjep_with_receipt(
        fake,
        tasks,
        config=TpjepRunConfig(model="offline-fake", template_class="gemma"),
    )
    assert [r.task_id for r in receipt.records] == list(EIGHT_TASK_IDS)
    assert receipt.summary.n_scheduled == 8
    assert receipt.metadata.local_concat_recipe.startswith("sha256_concat")
    assert receipt.metadata.manifest_recipe == "typellm_manifest_sha256"
    assert receipt.metadata.thinking is False
