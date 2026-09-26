"""Offline tests for PSAI metadata loader and Decision fixtures (#71)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from typevet.domain import compile_json_schema
from typevet.eval_psai import (
    CATEGORY_LABELS,
    OTHER_SUB_CATEGORY,
    PRIMARY_NOUL_NAME,
    SUB_CATEGORY_CHOICE_NAME,
    choice_sub_category,
    export_task,
    export_tasks,
    load_train_split,
    map_row,
    metadata_manifest,
    noul_requires_login,
    psai_state,
    validate_task_state,
)
from typevet.eval_psai_download import list_train_parquet_urls
from typevet.eval_psai_schema import (
    BANNED_STATE_KEYS,
    CONTRACT_ROW_MAX,
    CONTRACT_ROW_MIN,
    METADATA_DECISIONS_SCHEMA,
    METADATA_SCHEMA_VERSION,
)
from typevet.eval_psai_stream import (
    dedupe_shuffle_sample,
    iter_jsonl_rows,
    strip_heavy_fields,
)

FIXTURE_JSONL = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "metadata_smoke.jsonl"
)
FIXTURE_TEXT = FIXTURE_JSONL.read_text(encoding="utf-8")
VERSIONED_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "psai_metadata_decisions_schema_v1.json"
)
MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "psai_metadata_manifest_v1.yaml"
)


@pytest.mark.unit
def test_noul_requires_login_empty_is_false() -> None:
    assert noul_requires_login("") is False
    assert noul_requires_login("no") is False
    assert noul_requires_login("yes") is True


@pytest.mark.unit
def test_choice_sub_category_other_fallback() -> None:
    assert choice_sub_category([]) == OTHER_SUB_CATEGORY
    assert choice_sub_category(["Unknown label"]) == OTHER_SUB_CATEGORY
    assert choice_sub_category(["Search & Research"]) == "Search & Research"


@pytest.mark.unit
def test_strip_heavy_fields_drops_screenshots_without_decode() -> None:
    row = {"unique_data_id": "x", "screenshots": [{"bytes": b"\x00"}], "task_name": "t"}
    cleaned = strip_heavy_fields(row)
    assert "screenshots" not in cleaned
    assert cleaned["task_name"] == "t"


@pytest.mark.unit
def test_smoke_fixture_within_contract_bounds() -> None:
    line_count = sum(1 for line in FIXTURE_TEXT.splitlines() if line.strip())
    assert CONTRACT_ROW_MIN <= line_count <= CONTRACT_ROW_MAX


@pytest.mark.unit
def test_dedupe_shuffle_is_deterministic() -> None:
    duped = FIXTURE_TEXT + FIXTURE_TEXT
    a = load_train_split(jsonl_text=duped, limit=6, seed=0)
    b = load_train_split(jsonl_text=duped, limit=6, seed=0)
    assert [e.unique_data_id for e in a] == [e.unique_data_id for e in b]
    assert len(a) == 6


@pytest.mark.unit
def test_map_row_from_fixture_line() -> None:
    first = json.loads(FIXTURE_TEXT.splitlines()[0])
    example = map_row(first)
    assert example.category in CATEGORY_LABELS
    assert example.requires_login is False


@pytest.mark.unit
def test_validate_task_state_rejects_leakage() -> None:
    state = psai_state("Do something.")
    validate_task_state(state)
    with pytest.raises(ValueError, match="leakage"):
        validate_task_state({**state, "category": "BROWSER_TASK"})


@pytest.mark.unit
def test_export_task_jevbench_shape() -> None:
    first = json.loads(FIXTURE_TEXT.splitlines()[0])
    example = map_row(first)
    task = export_task(example)
    assert task["family"] == "psai"
    assert task["expected"]["category"] == example.category
    assert task["expected"][PRIMARY_NOUL_NAME] is example.requires_login
    assert task["state"] == {"task_name": example.task_name}
    assert SUB_CATEGORY_CHOICE_NAME in task["expected"]
    assert "category" not in task["state"]
    assert len(task["questions"]) == 7


@pytest.mark.unit
def test_export_tasks_from_fixture() -> None:
    rows = load_train_split(jsonl_text=FIXTURE_TEXT, limit=4, seed=0)
    tasks = export_tasks(rows)
    assert len(tasks) == 4
    for task in tasks:
        validate_task_state(task["state"])


@pytest.mark.unit
def test_versioned_json_fixture_matches_python_schema() -> None:
    on_disk = json.loads(VERSIONED_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == METADATA_DECISIONS_SCHEMA
    assert METADATA_SCHEMA_VERSION == "1"


@pytest.mark.unit
def test_metadata_manifest_yaml_matches_python() -> None:
    on_disk = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk == metadata_manifest()


@pytest.mark.unit
def test_metadata_schema_compiles() -> None:
    decisions = compile_json_schema(METADATA_DECISIONS_SCHEMA)
    assert len(decisions) == 7
    names = {decision.name for decision in decisions}
    assert PRIMARY_NOUL_NAME in names
    assert "category" in names


@pytest.mark.unit
def test_list_train_parquet_urls_mock_client() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host != "datasets-server.huggingface.co":
                return httpx.Response(404)
            return httpx.Response(
                200,
                json={
                    "parquet_files": [
                        {
                            "url": "https://example.test/0003.parquet",
                            "filename": "0003.parquet",
                        }
                    ]
                },
            )

    client = httpx.Client(transport=FakeTransport())
    urls = list_train_parquet_urls(client=client)
    assert urls == ["https://example.test/0003.parquet"]


@pytest.mark.unit
def test_banned_state_keys_include_metadata_fields() -> None:
    assert "screenshots" in BANNED_STATE_KEYS
    assert "subCategory" in BANNED_STATE_KEYS


@pytest.mark.unit
def test_dedupe_shuffle_sample_respects_limit() -> None:

    sampled = dedupe_shuffle_sample(iter_jsonl_rows(FIXTURE_TEXT), limit=3, seed=1)
    assert len(sampled) == 3
