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
    - [typevet_evals.datasets.psai_evidence_pilot][]: the loader under test
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest

from typevet_evals.datasets.psai_evidence_pilot import (
    ManifestError,
    load_manifest,
)

pytestmark = pytest.mark.contract

PSAI_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "psai"
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


def _task_words_in_claim(m: Manifest) -> None:
    # Exactly 4 consecutive words of this row's own task_name.
    row = m["rows"][PROMPT + 6]
    assert row["task_name"].startswith("Synthetic task 7 for the ")
    row["model_input"]["claim"] = "The page says synthetic task 7 for users."


def _bad_label(m: Manifest) -> None:
    m["rows"][DEV]["gold_label"] = "maybe"


def _case_id_is_source_id(m: Manifest) -> None:
    m["rows"][DEV]["case_id"] = m["rows"][DEV]["unique_data_id"]


def _case_id_twice(m: Manifest) -> None:
    m["rows"][FINAL + 1]["case_id"] = m["rows"][DEV]["case_id"]


def _gold_label_word_in_claim(m: Manifest) -> None:
    row = m["rows"][DEV]
    assert row["gold_label"] == "supported"
    row["model_input"]["claim"] = "The claim is SUPPORTED by the page."


def _gold_label_spaced_in_claim(m: Manifest) -> None:
    row = m["rows"][DEV + 20]
    assert row["gold_label"] == "insufficient_evidence"
    row["model_input"]["claim"] = "The page gives Insufficient Evidence here."


def _gold_label_underscored_in_claim(m: Manifest) -> None:
    row = m["rows"][DEV + 20]
    row["model_input"]["claim"] = "Answer: insufficient_evidence."


def _gold_label_hyphenated_in_claim(m: Manifest) -> None:
    row = m["rows"][DEV + 20]
    assert row["gold_label"] == "insufficient_evidence"
    row["model_input"]["claim"] = "The page gives insufficient-evidence here."


def _gold_label_double_spaced_in_claim(m: Manifest) -> None:
    row = m["rows"][DEV + 20]
    row["model_input"]["claim"] = "The page gives insufficient  evidence here."


def _gold_label_joined_to_snake_word(m: Manifest) -> None:
    row = m["rows"][DEV]
    assert row["gold_label"] == "supported"
    row["model_input"]["claim"] = "The claim is supported_by the page."


def _gold_label_non_breaking_hyphen(m: Manifest) -> None:
    row = m["rows"][DEV + 20]
    assert row["gold_label"] == "insufficient_evidence"
    row["model_input"]["claim"] = "The page gives insufficient\u2011evidence here."


def _gold_label_unicode_hyphen(m: Manifest) -> None:
    row = m["rows"][DEV + 20]
    assert row["gold_label"] == "insufficient_evidence"
    row["model_input"]["claim"] = "The page gives insufficient\u2010evidence here."


def _gold_label_soft_hyphen_inside(m: Manifest) -> None:
    row = m["rows"][DEV]
    assert row["gold_label"] == "supported"
    row["model_input"]["claim"] = "The claim is sup\u00adported by the page."


def _gold_label_fullwidth(m: Manifest) -> None:
    row = m["rows"][DEV]
    assert row["gold_label"] == "supported"
    row["model_input"]["claim"] = (
        "The claim is \uff53\uff55\uff50\uff50\uff4f\uff52\uff54\uff45\uff44."
    )


def _rejected_is_row_id(m: Manifest) -> None:
    m["rejected"][0]["unique_data_id"] = m["rows"][PROMPT + 2]["unique_data_id"]


def _rejected_twice(m: Manifest) -> None:
    m["rejected"][1]["unique_data_id"] = m["rejected"][0]["unique_data_id"]


def _rejected_in_fixtures(m: Manifest) -> None:
    m["rejected"][1]["unique_data_id"] = FIXTURE_UID


def _task_without_claim_axis(m: Manifest) -> None:
    # dev-01 loses its claim_axis row; dev-08 gains a second one.
    m["rows"][DEV + 8]["unique_data_id"] = m["rows"][DEV + 7]["unique_data_id"]


def _task_with_two_image_rows(m: Manifest) -> None:
    # dev-01 gains the crop row of dev-08 beside its own swap row.
    m["rows"][DEV + 23]["unique_data_id"] = m["rows"][DEV]["unique_data_id"]


def _seven_tasks_in_split(m: Manifest) -> None:
    # Fold task dev-01 (no donor refers to it) into dev-02: 7 distinct IDs.
    for offset in (0, 8, 16):
        m["rows"][DEV + offset]["unique_data_id"] = m["rows"][DEV + 1]["unique_data_id"]


MUTATIONS: list[tuple[str, FunctionType]] = [
    ("case_id_unique", _case_id_twice),
    ("gold_label_leak", _gold_label_word_in_claim),
    ("gold_label_leak", _gold_label_spaced_in_claim),
    ("gold_label_leak", _gold_label_underscored_in_claim),
    ("gold_label_leak", _gold_label_hyphenated_in_claim),
    ("gold_label_leak", _gold_label_double_spaced_in_claim),
    ("gold_label_leak", _gold_label_joined_to_snake_word),
    ("gold_label_leak", _gold_label_non_breaking_hyphen),
    ("gold_label_leak", _gold_label_unicode_hyphen),
    ("gold_label_leak", _gold_label_soft_hyphen_inside),
    ("gold_label_leak", _gold_label_fullwidth),
    ("rejected_items", _rejected_is_row_id),
    ("rejected_items", _rejected_twice),
    ("rejected_items", _rejected_in_fixtures),
    ("task_inventory", _task_without_claim_axis),
    ("task_inventory", _task_with_two_image_rows),
    ("task_inventory", _seven_tasks_in_split),
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
    ("input_leak", _task_words_in_claim),
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


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_task_without_claim_axis, "dev synthetic-dev-01 has"),
        (_task_with_two_image_rows, "dev synthetic-dev-01 has"),
        (_seven_tasks_in_split, "dev has 7 tasks"),
        (_rejected_is_row_id, "is a row ID"),
        (_rejected_twice, "twice"),
        (_rejected_in_fixtures, f"rejected {FIXTURE_UID} is in"),
    ],
    ids=lambda value: getattr(value, "__name__", "detail").lstrip("_"),
)
def test_each_clause_names_its_own_failure(
    tmp_path: Path, mutate: Callable[[Manifest], None], expected: str
):
    """Each clause of a multi-clause rule reports its own cause."""
    manifest = _valid()
    mutate(manifest)

    with pytest.raises(ManifestError) as caught:
        load_manifest(_write(tmp_path, manifest), pilot_root=PILOT_ROOT)

    assert expected in str(caught.value)


def test_label_inside_a_longer_word_is_not_a_gold_label_leak(tmp_path: Path):
    """``unsupported`` does not hold the whole word ``supported``."""
    manifest = _valid()
    manifest["rows"][DEV]["model_input"]["claim"] = "The button is unsupported."

    loaded = load_manifest(_write(tmp_path, manifest), pilot_root=PILOT_ROOT)

    assert loaded.rows[DEV]["model_input"]["claim"] == "The button is unsupported."


@pytest.mark.parametrize(
    ("case", "claim"),
    [
        (DEV, "The claim is supportedly true."),
        (PROMPT + 6, "The page says synthetic task 7 is done."),
    ],
    ids=["label-prefix-of-longer-word", "three-task-words"],
)
def test_near_miss_claim_loads(tmp_path: Path, case: int, claim: str):
    """A label inside a longer word and a 3-word task run are not leaks."""
    manifest = _valid()
    manifest["rows"][case]["model_input"]["claim"] = claim

    loaded = load_manifest(_write(tmp_path, manifest), pilot_root=PILOT_ROOT)

    assert loaded.rows[case]["model_input"]["claim"] == claim
