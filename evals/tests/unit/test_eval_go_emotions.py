"""Offline tests for go_emotions simplified loader (#79)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from typevet.domain import compile_json_schema
from typevet_evals.datasets.go_emotions import (
    BANNED_STATE_KEYS,
    CHOICE_LABELS,
    CONFIG,
    EMOTION_CHOICE_SCHEMA,
    NEUTRAL_LABEL,
    PRIMARY_CHOICE_NAME,
    PRUNED_LABELS,
    GoEmotionsExample,
    assert_state_keys_allowed,
    balanced_sample,
    build_state,
    iter_train_rows,
    labels_after_neutral_rule,
    load_train_split,
    map_examples,
    map_row,
    strict_gold_label,
    try_map_row,
)
from typevet_evals.datasets.go_emotions_download import download_train_jsonl

FIXTURE_JSONL = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "go_emotions"
    / "simplified_train_subset.jsonl"
)
FIXTURE_TEXT = FIXTURE_JSONL.read_text(encoding="utf-8")
VERSIONED_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals"
    / "fixtures"
    / "go_emotions_emotion_choice_schema_v1.json"
)


@pytest.mark.unit
def test_choice_labels_count_and_prune() -> None:
    assert len(CHOICE_LABELS) == 24
    assert (
        frozenset(
            {"grief", "nervousness", "pride", "relief"},
        )
        == PRUNED_LABELS
    )
    assert NEUTRAL_LABEL in CHOICE_LABELS


@pytest.mark.unit
def test_labels_after_neutral_rule_strips_co_label() -> None:
    assert labels_after_neutral_rule(["neutral", "joy"]) == ["joy"]
    assert labels_after_neutral_rule(["neutral"]) == ["neutral"]


@pytest.mark.unit
def test_strict_gold_label_drops_multi_label() -> None:
    assert strict_gold_label(["desire", "optimism"]) is None
    assert strict_gold_label(["anger"]) == "anger"


@pytest.mark.unit
def test_strict_gold_label_drops_pruned_only() -> None:
    assert strict_gold_label(["grief"]) is None
    assert strict_gold_label(["pride"]) is None


@pytest.mark.unit
def test_build_state_and_banned_keys() -> None:
    state = build_state("hello")
    assert state == {"text": "hello"}
    assert_state_keys_allowed(state)
    assert "labels" in BANNED_STATE_KEYS


@pytest.mark.unit
def test_assert_state_keys_allowed_rejects_labels() -> None:
    with pytest.raises(ValueError, match="banned keys"):
        assert_state_keys_allowed({"text": "x", "labels": ["joy"]})


@pytest.mark.unit
def test_try_map_row_drops_ambiguous() -> None:
    record = {"id": "x", "text": "t", "labels": ["joy", "sadness"]}
    assert try_map_row(record) is None


@pytest.mark.unit
def test_map_row_from_minimal_record() -> None:
    record = {"id": "abc", "text": "nice!", "labels": ["admiration"]}
    ex = map_row(record)
    assert ex == GoEmotionsExample(
        row_id="abc",
        state={"text": "nice!"},
        choice_label="admiration",
    )
    assert ex.config == CONFIG


@pytest.mark.unit
def test_try_map_row_accepts_label_indices() -> None:
    record = {"id": "i", "text": "t", "labels": [2]}
    ex = try_map_row(record)
    assert ex is not None
    assert ex.choice_label == "anger"


@pytest.mark.unit
def test_iter_train_rows_rejects_bad_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        list(iter_train_rows("{not json}\n"))


@pytest.mark.unit
def test_load_fixture_subset() -> None:
    rows = load_train_split(jsonl_text=FIXTURE_TEXT)
    assert len(rows) == 12
    assert all(r.choice_label in CHOICE_LABELS for r in rows)


@pytest.mark.unit
def test_map_examples_skips_dropped_rows_in_fixture() -> None:
    mixed = FIXTURE_TEXT + json.dumps(
        {
            "id": "drop1",
            "text": "multi",
            "labels": ["joy", "love"],
        },
        ensure_ascii=False,
    )
    rows = map_examples(iter_train_rows(mixed))
    assert len(rows) == 12


@pytest.mark.unit
def test_balanced_sample_from_fixture() -> None:
    all_rows = load_train_split(jsonl_text=FIXTURE_TEXT)
    sample = balanced_sample(all_rows, limit=24, seed=0)
    assert len(sample) <= 24
    assert {r.choice_label for r in sample}.issubset(set(CHOICE_LABELS))


@pytest.mark.unit
def test_load_train_split_limit() -> None:
    rows = load_train_split(jsonl_text=FIXTURE_TEXT, limit=2)
    assert len(rows) == 2
    first = json.loads(FIXTURE_TEXT.splitlines()[0])
    assert rows[0].row_id == first["id"]


@pytest.mark.unit
def test_emotion_choice_schema_compiles() -> None:
    decisions = compile_json_schema(EMOTION_CHOICE_SCHEMA)
    assert len(decisions) == 1
    assert decisions[0].name == PRIMARY_CHOICE_NAME
    assert decisions[0].syntax == "Choice"
    assert decisions[0].choices == CHOICE_LABELS


@pytest.mark.unit
def test_versioned_schema_matches_module() -> None:
    on_disk = json.loads(VERSIONED_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == EMOTION_CHOICE_SCHEMA


@pytest.mark.unit
def test_download_train_jsonl_uses_injected_client() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host != "datasets-server.huggingface.co":
                return httpx.Response(404)
            if request.url.path.endswith("/info"):
                payload = {
                    "dataset_info": {CONFIG: {"splits": {"train": {"num_examples": 1}}}}
                }
                return httpx.Response(200, json=payload)
            row = json.loads(FIXTURE_TEXT.splitlines()[0])
            hub_row = {
                "id": row["id"],
                "text": row["text"],
                "labels": [27 if row["labels"] == ["neutral"] else 2],
            }
            return httpx.Response(200, json={"rows": [{"row": hub_row}]})

    client = httpx.Client(transport=FakeTransport())
    text = download_train_jsonl(client=client)
    rows = load_train_split(jsonl_text=text)
    assert len(rows) == 1
    assert rows[0].choice_label == "neutral"


@pytest.mark.unit
def test_load_train_split_downloads_when_jsonl_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(*, client: httpx.Client | None = None) -> str:
        assert client is None
        return FIXTURE_TEXT

    monkeypatch.setattr(
        "typevet_evals.datasets.go_emotions.download_train_jsonl",
        fake_download,
    )
    rows = load_train_split(limit=1)
    assert len(rows) == 1
