"""Offline tests for BoolQ loader and answer Noul fixture (#77)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from typevet.domain import compile_json_schema
from typevet.eval_boolq import (
    BANNED_STATE_KEYS,
    BOOLQ_ANSWER_NOUL_SCHEMA,
    BOOLQ_ANSWER_NOUL_SCHEMA_VERSION,
    CONTRACT_ROW_MAX,
    CONTRACT_ROW_MIN,
    DEFAULT_TIER_SEED,
    NOUL_LABELS,
    PRIMARY_NOUL_NAME,
    QUESTION_MARKER,
    TIER_A_LIMIT,
    TIER_B_LIMIT,
    BoolQExample,
    balanced_sample,
    boolq_state,
    download_validation_jsonl,
    export_task,
    export_tasks,
    iter_validation_rows,
    load_tier,
    load_tier_a,
    load_tier_b,
    load_validation_split,
    map_row,
    noul_label_for_answer,
    serialize_boolq_state,
    tier_limit,
    tier_manifest,
    validate_task_state,
)

FIXTURE_JSONL = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "boolq"
    / "validation_smoke.jsonl"
)
FIXTURE_TEXT = FIXTURE_JSONL.read_text(encoding="utf-8")
VERSIONED_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "boolq_answer_noul_schema_v1.json"
)
TIER_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "boolq_tier_manifest_v1.yaml"
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flag", "label"),
    [(True, "yes"), (False, "no")],
)
def test_noul_label_for_answer(flag: bool, label: str) -> None:
    assert noul_label_for_answer(flag) == label
    assert NOUL_LABELS == ("no", "yes")


@pytest.mark.unit
def test_map_row_sets_fields() -> None:
    ex = map_row("p", "q?", True, idx=7)
    assert ex == BoolQExample(passage="p", question="q?", label="yes", idx=7)


@pytest.mark.unit
def test_boolq_state_and_serializer() -> None:
    state = boolq_state("passage text", "question text?")
    assert state == {"passage": "passage text", "question": "question text?"}
    validate_task_state(state)
    serialized = serialize_boolq_state("passage text", "question text?")
    assert "--- passage ---" in serialized
    assert QUESTION_MARKER in serialized
    assert "passage text" in serialized


@pytest.mark.unit
def test_validate_task_state_rejects_banned_keys() -> None:
    for key in BANNED_STATE_KEYS:
        with pytest.raises(ValueError, match="leakage"):
            validate_task_state({key: "x", "passage": "p", "question": "q"})


@pytest.mark.unit
def test_iter_validation_rows_parses_fixture() -> None:
    rows = list(iter_validation_rows(FIXTURE_TEXT))
    assert len(rows) == 12
    assert rows[0][3] == 0


@pytest.mark.unit
def test_iter_validation_rows_rejects_bad_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        list(iter_validation_rows("{not json}\n"))


@pytest.mark.unit
def test_iter_validation_rows_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="passage"):
        list(iter_validation_rows('{"answer":true}\n'))


@pytest.mark.unit
def test_smoke_fixture_within_contract_bounds() -> None:
    line_count = sum(1 for line in FIXTURE_TEXT.splitlines() if line.strip())
    assert CONTRACT_ROW_MIN <= line_count <= CONTRACT_ROW_MAX


@pytest.mark.unit
def test_load_fixture_subset_unbalanced() -> None:
    rows = load_validation_split(jsonl_text=FIXTURE_TEXT)
    assert len(rows) == 12
    assert sum(1 for r in rows if r.label == "yes") == 6


@pytest.mark.unit
def test_balanced_sample_from_fixture() -> None:
    all_rows = load_validation_split(jsonl_text=FIXTURE_TEXT)
    sample = balanced_sample(all_rows, limit=4, seed=0)
    assert len(sample) == 4
    assert sum(1 for r in sample if r.label == "yes") == 2
    assert sum(1 for r in sample if r.label == "no") == 2


@pytest.mark.unit
def test_load_validation_split_balanced_flag() -> None:
    rows = load_validation_split(
        jsonl_text=FIXTURE_TEXT, balanced=True, limit=4, seed=1
    )
    assert len(rows) == 4
    assert {r.label for r in rows} == {"yes", "no"}


@pytest.mark.unit
def test_tier_limits_and_helpers() -> None:
    assert tier_limit("A") == TIER_A_LIMIT == 24
    assert tier_limit("B") == TIER_B_LIMIT == 256
    rows_a = load_tier_a(jsonl_text=FIXTURE_TEXT, seed=0)
    assert len(rows_a) == 12
    assert sum(1 for r in rows_a if r.label == "yes") == 6
    rows_b = load_tier_b(jsonl_text=FIXTURE_TEXT, seed=0)
    assert len(rows_b) == len(rows_a)
    assert load_tier("A", jsonl_text=FIXTURE_TEXT, seed=0) == rows_a


@pytest.mark.unit
def test_tier_manifest_matches_yaml() -> None:
    on_disk = yaml.safe_load(TIER_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert tier_manifest() == on_disk
    assert on_disk["default_seed"] == DEFAULT_TIER_SEED


@pytest.mark.unit
def test_answer_noul_schema_compiles() -> None:
    decisions = compile_json_schema(BOOLQ_ANSWER_NOUL_SCHEMA)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.name == PRIMARY_NOUL_NAME
    assert decision.syntax == "Choice"
    assert decision.choices == NOUL_LABELS
    assert decision.return_probabilities is True


@pytest.mark.unit
def test_versioned_json_fixture_matches_python_schema() -> None:
    on_disk = json.loads(VERSIONED_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == BOOLQ_ANSWER_NOUL_SCHEMA
    assert BOOLQ_ANSWER_NOUL_SCHEMA_VERSION == "1"


@pytest.mark.unit
def test_export_task_jevbench_shape() -> None:
    example = map_row("Passage.", "Is it true?", True, idx=3)
    task = export_task(example)
    assert task["id"] == "boolq-validation-3"
    assert task["family"] == "boolq"
    assert task["split"] == "validation"
    assert task["expected"] == {"answer": "yes"}
    assert task["state"] == {"passage": "Passage.", "question": "Is it true?"}
    assert "answer" not in task["state"]
    assert task["questions"][0]["labels"] == ["no", "yes"]
    assert task["provenance"]["dataset_id"] == "google/boolq"


@pytest.mark.unit
def test_export_tasks_from_fixture() -> None:
    rows = load_validation_split(jsonl_text=FIXTURE_TEXT, limit=4, balanced=True)
    tasks = export_tasks(rows)
    assert len(tasks) == 4
    for task in tasks:
        validate_task_state(task["state"])


@pytest.mark.unit
def test_download_validation_jsonl_uses_injected_client() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host != "datasets-server.huggingface.co":
                return httpx.Response(404)
            if "/info" in str(request.url):
                body = {
                    "dataset_info": {
                        "default": {
                            "splits": {"validation": {"num_examples": 2}},
                        }
                    }
                }
                return httpx.Response(200, json=body)
            return httpx.Response(
                200,
                json={
                    "rows": [
                        {
                            "row": {
                                "passage": "p1",
                                "question": "q1?",
                                "answer": True,
                            }
                        },
                        {
                            "row": {
                                "passage": "p2",
                                "question": "q2?",
                                "answer": False,
                            }
                        },
                    ]
                },
            )

    client = httpx.Client(transport=FakeTransport())
    jsonl_text = download_validation_jsonl(client=client, page_size=2)
    rows = load_validation_split(jsonl_text=jsonl_text)
    assert len(rows) == 2
    assert rows[0].label == "yes"
    assert rows[0].idx == 0
    assert rows[1].label == "no"
    assert rows[1].idx == 1


@pytest.mark.unit
def test_load_validation_split_downloads_when_jsonl_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(**kwargs: object) -> str:
        return FIXTURE_TEXT

    monkeypatch.setattr(
        "typevet.eval_boolq.download_validation_jsonl",
        fake_download,
    )
    rows = load_validation_split(limit=2)
    assert len(rows) == 2
