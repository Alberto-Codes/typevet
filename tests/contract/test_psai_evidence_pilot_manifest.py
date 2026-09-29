"""Offline contract: PSAI evidence pilot v1 manifest schema and rules (#192).

One synthetic manifest (72 rows, no screenshots) must load. Each mutated copy
breaks exactly one rule and must raise ``ManifestError`` that names that rule.
These tests prove manifest bookkeeping only. They say nothing about label
quality or any model.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_psai_evidence_pilot_manifest.py
    ```

See Also:
    - [typevet.evaluation.datasets.psai_evidence_pilot][]: the loader under test
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest

from typevet.evaluation.datasets.psai_evidence_pilot import (
    ManifestError,
    load_manifest,
)

pytestmark = pytest.mark.contract

PSAI_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "psai"
PILOT_ROOT = PSAI_ROOT / "evidence_pilot_v1"
SYNTHETIC = PILOT_ROOT / "synthetic" / "manifest.json"
# Row blocks per split: 0-7 baseline, 8-15 claim_axis, 16-19 swap, 20-23 crop.
DEV, PROMPT, FINAL = 0, 24, 48
FIXTURE_UID = "cmbcvas3900v3yl0x93t0oy24"  # in tests/fixtures/psai/metadata_smoke

Manifest = dict[str, Any]


def _valid() -> Manifest:
    return json.loads(SYNTHETIC.read_text(encoding="utf-8"))


def _host_from_other_split(m: Manifest) -> None:
    m["rows"][DEV + 2]["host"] = m["rows"][PROMPT + 2]["host"]


def _donor_from_other_split(m: Manifest) -> None:
    donor_row = m["rows"][PROMPT + 5]
    m["rows"][DEV + 16]["donor"] = {
        "unique_data_id": donor_row["unique_data_id"],
        "host": donor_row["host"],
        "screenshot_index": donor_row["screenshot_index"],
        "png_sha256": donor_row["png_sha256"],
    }


def _donor_shares_host(m: Manifest) -> None:
    row = m["rows"][DEV + 18]
    row["donor"]["host"] = row["host"]


def _fixture_uid(m: Manifest) -> None:
    m["rows"][DEV + 21]["unique_data_id"] = FIXTURE_UID


def _too_few_insufficient(m: Manifest) -> None:
    row = m["rows"][DEV + 20]
    assert row["gold_label"] == "insufficient_evidence"
    row["gold_label"] = "contradicted"
    row["votes"]["human"] = "contradicted"


def _swap_becomes_crop(m: Manifest) -> None:
    row = m["rows"][DEV + 17]
    row["row_kind"] = "crop"
    row["donor"] = None


def _drop_baseline(m: Manifest) -> None:
    del m["rows"][PROMPT + 3]


def _final_without_human(m: Manifest) -> None:
    row = m["rows"][FINAL + 1]
    row["label_source"] = "gemma_draft"
    row["votes"]["human"] = None


def _crop_without_human(m: Manifest) -> None:
    row = m["rows"][DEV + 21]
    row["label_source"] = "gemma_draft"
    row["votes"]["human"] = None
    row["votes"]["gemma_draft"] = row["gold_label"]
    row["votes"]["qwen_dom"] = copy.deepcopy(m["rows"][DEV]["votes"]["qwen_dom"])


def _human_vote_not_gold(m: Manifest) -> None:
    m["rows"][FINAL + 9]["votes"]["human"] = "supported"


def _qwen_minority(m: Manifest) -> None:
    runs = m["rows"][DEV + 3]["votes"]["qwen_dom"]["run_labels"]
    runs[0] = runs[1] = "contradicted"


def _gemma_disagrees(m: Manifest) -> None:
    m["rows"][PROMPT + 4]["votes"]["gemma_draft"] = "contradicted"


def _construction_disagrees(m: Manifest) -> None:
    m["rows"][DEV + 9]["votes"]["construction"] = "supported"


def _named_source_without_vote(m: Manifest) -> None:
    m["rows"][PROMPT + 1]["label_source"] = "construction"


def _qwen_missing(m: Manifest) -> None:
    m["rows"][DEV + 10]["votes"]["qwen_dom"] = None


def _host_in_claim(m: Manifest) -> None:
    row = m["rows"][DEV + 4]
    row["model_input"]["claim"] = f"The page at {row['host'].upper()} shows a form."


def _task_name_in_claim(m: Manifest) -> None:
    row = m["rows"][PROMPT + 6]
    row["model_input"]["claim"] = f"Done: {row['task_name']}"


def _bad_label(m: Manifest) -> None:
    m["rows"][DEV]["gold_label"] = "maybe"


def _case_id_is_source_id(m: Manifest) -> None:
    m["rows"][DEV]["case_id"] = m["rows"][DEV]["unique_data_id"]


MUTATIONS: list[tuple[str, FunctionType]] = [
    ("split_isolation", _host_from_other_split),
    ("split_isolation", _donor_from_other_split),
    ("split_isolation", _donor_shares_host),
    ("fixture_overlap", _fixture_uid),
    ("gold_per_class", _too_few_insufficient),
    ("split_counts", _swap_becomes_crop),
    ("split_counts", _drop_baseline),
    ("human_decision", _final_without_human),
    ("human_decision", _crop_without_human),
    ("human_decision", _human_vote_not_gold),
    ("verified_agreement", _qwen_minority),
    ("verified_agreement", _gemma_disagrees),
    ("verified_agreement", _construction_disagrees),
    ("verified_agreement", _qwen_missing),
    ("verified_agreement", _named_source_without_vote),
    ("input_leak", _host_in_claim),
    ("input_leak", _task_name_in_claim),
    ("schema", _bad_label),
    ("schema", _case_id_is_source_id),
]


def _write(tmp_path: Path, manifest: Manifest) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_synthetic_manifest_loads_72_rows_and_keeps_rejects_apart():
    """The checked-in synthetic manifest loads and rejected items stay apart."""
    manifest = load_manifest(SYNTHETIC)

    assert len(manifest.rows) == 72
    assert len(manifest.rejected) == 2
    assert all(item["reason_codes"] for item in manifest.rejected)
    assert {row["split"] for row in manifest.rows} == {
        "dev",
        "prompt_selection",
        "final",
    }
    assert len(manifest.split_rows("final")) == 24


def test_copied_valid_manifest_loads_with_explicit_pilot_root(tmp_path: Path):
    """An unmutated copy outside the fixture tree loads with an explicit root."""
    manifest = load_manifest(_write(tmp_path, _valid()), pilot_root=PILOT_ROOT)

    assert len(manifest.rows) == 72


@pytest.mark.parametrize(
    ("rule", "mutate"),
    MUTATIONS,
    ids=[f"{rule}-{fn.__name__.lstrip('_')}" for rule, fn in MUTATIONS],
)
def test_each_mutation_raises_for_its_own_rule(
    tmp_path: Path, rule: str, mutate: Callable[[Manifest], None]
):
    """Each single-rule mutation raises an error that names that rule."""
    manifest = _valid()
    mutate(manifest)

    with pytest.raises(ManifestError) as caught:
        load_manifest(_write(tmp_path, manifest), pilot_root=PILOT_ROOT)

    assert caught.value.rule == rule
    assert str(caught.value).startswith(f"{rule}:")


def test_manifest_outside_a_pilot_root_needs_an_explicit_root(tmp_path: Path):
    """Without an ``evidence_pilot_v1`` ancestor the caller must name the root."""
    with pytest.raises(ManifestError) as caught:
        load_manifest(_write(tmp_path, _valid()))

    assert caught.value.rule == "pilot_root"
