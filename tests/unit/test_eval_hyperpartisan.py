"""Offline tests for hyperpartisan byarticle loader and Noul fixture (#81)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from typevet.domain import compile_json_schema
from typevet.eval_hyperpartisan import (
    BANNED_STATE_KEYS,
    BODY_MAX_CHARS,
    CONTRACT_ROW_MAX,
    CONTRACT_ROW_MIN,
    DEFAULT_HOLDOUT_SEED,
    EXCLUDED_SPLIT,
    HOLDOUT_FRACTION,
    HYPERPARTISAN_NOUL_SCHEMA,
    HYPERPARTISAN_NOUL_SCHEMA_VERSION,
    PRIMARY_NOUL_NAME,
    clean_html,
    export_task,
    export_tasks,
    holdout_manifest,
    hyperpartisan_state,
    iter_byarticle_rows,
    load_byarticle_pool,
    load_holdout_split,
    load_train_split,
    map_row,
    partition_rows,
    stratified_holdout_ids,
    truncate_body,
    validate_task_state,
)

FIXTURE_JSONL = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "hyperpartisan"
    / "byarticle_smoke.jsonl"
)
FIXTURE_TEXT = FIXTURE_JSONL.read_text(encoding="utf-8")
VERSIONED_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "hyperpartisan_hyperpartisan_noul_schema_v1.json"
)
HOLDOUT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "hyperpartisan_holdout_manifest_v1.yaml"
)


@pytest.mark.unit
def test_clean_html_strips_tags_and_entities() -> None:
    assert clean_html("<p>Hello <b>world</b>&nbsp;!</p>") == "Hello world !"


@pytest.mark.unit
def test_truncate_body_respects_cap() -> None:
    long_text = "a" * (BODY_MAX_CHARS + 10)
    trimmed = truncate_body(long_text)
    assert len(trimmed) == BODY_MAX_CHARS
    assert trimmed.endswith("...")


@pytest.mark.unit
def test_map_row_cleans_body() -> None:
    ex = map_row("id-1", "Title", "<i>Body</i>", True, partition="holdout")
    assert ex.body == "Body"
    assert ex.title == "Title"
    assert ex.partition == "holdout"


@pytest.mark.unit
def test_smoke_fixture_within_contract_bounds() -> None:
    line_count = sum(1 for line in FIXTURE_TEXT.splitlines() if line.strip())
    assert CONTRACT_ROW_MIN <= line_count <= CONTRACT_ROW_MAX


@pytest.mark.unit
def test_iter_byarticle_rows_parses_fixture() -> None:
    rows = list(iter_byarticle_rows(FIXTURE_TEXT))
    assert len(rows) == 12
    assert sum(1 for *_, flag in rows if flag) == 6


@pytest.mark.unit
def test_iter_byarticle_rows_rejects_bypublisher() -> None:
    bad = json.dumps(
        {
            "article_id": "x",
            "title": "t",
            "body": "b",
            "hyperpartisan": False,
            "split": EXCLUDED_SPLIT,
        }
    )
    with pytest.raises(ValueError, match="bypublisher"):
        list(iter_byarticle_rows(bad + "\n"))


@pytest.mark.unit
def test_stratified_holdout_is_deterministic_and_balanced() -> None:
    parsed = list(iter_byarticle_rows(FIXTURE_TEXT))
    ids_a = stratified_holdout_ids(parsed, seed=0)
    ids_b = stratified_holdout_ids(parsed, seed=0)
    assert ids_a == ids_b
    assert 2 <= len(ids_a) <= 4


@pytest.mark.unit
def test_partition_rows_disjoint_and_cover_pool() -> None:
    train, holdout = partition_rows(FIXTURE_TEXT, seed=DEFAULT_HOLDOUT_SEED)
    train_ids = {e.article_id for e in train}
    holdout_ids = {e.article_id for e in holdout}
    assert train_ids.isdisjoint(holdout_ids)
    assert len(train) + len(holdout) == 12
    assert all(e.partition == "holdout" for e in holdout)


@pytest.mark.unit
def test_load_splits_match_partition_rows() -> None:
    train, holdout = partition_rows(FIXTURE_TEXT, seed=0)
    assert load_train_split(jsonl_text=FIXTURE_TEXT, seed=0) == train
    assert load_holdout_split(jsonl_text=FIXTURE_TEXT, seed=0) == holdout
    assert len(load_byarticle_pool(jsonl_text=FIXTURE_TEXT, partition="all")) == 12


@pytest.mark.unit
def test_holdout_manifest_matches_yaml() -> None:
    on_disk = yaml.safe_load(HOLDOUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert holdout_manifest() == on_disk
    assert on_disk["holdout_fraction"] == HOLDOUT_FRACTION


@pytest.mark.unit
def test_hyperpartisan_noul_schema_compiles() -> None:
    decisions = compile_json_schema(HYPERPARTISAN_NOUL_SCHEMA)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.name == PRIMARY_NOUL_NAME
    assert decision.syntax == "Bool"
    assert decision.return_probabilities is True


@pytest.mark.unit
def test_versioned_json_fixture_matches_python_schema() -> None:
    on_disk = json.loads(VERSIONED_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == HYPERPARTISAN_NOUL_SCHEMA
    assert HYPERPARTISAN_NOUL_SCHEMA_VERSION == "1"


@pytest.mark.unit
def test_validate_task_state_rejects_banned_keys() -> None:
    for key in BANNED_STATE_KEYS:
        with pytest.raises(ValueError, match="leakage"):
            validate_task_state({key: "x", "title": "t", "body": "b"})


@pytest.mark.unit
def test_export_task_jevbench_shape() -> None:
    example = map_row("hp-smoke-007", "Title", "Body", True, partition="holdout")
    task = export_task(example)
    assert task["id"] == "hyperpartisan-holdout-hp-smoke-007"
    assert task["split"] == "holdout"
    assert task["expected"] == {"hyperpartisan": True}
    assert task["state"] == hyperpartisan_state("Title", "Body")
    assert "hyperpartisan" not in task["state"]
    assert task["questions"][0]["name"] == PRIMARY_NOUL_NAME


@pytest.mark.unit
def test_export_tasks_from_holdout_fixture() -> None:
    rows = load_holdout_split(jsonl_text=FIXTURE_TEXT, seed=0)
    tasks = export_tasks(rows)
    assert tasks
    for task in tasks:
        validate_task_state(task["state"])
