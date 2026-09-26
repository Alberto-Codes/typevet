"""Offline tests for PubMedQA pqa_labeled loader (#78)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from typevet.domain import compile_json_schema
from typevet.eval_pubmedqa import (
    ANSWER_CHOICE_SCHEMA,
    BANNED_STATE_KEYS,
    CHOICE_LABELS,
    EXCLUDED_CONFIGS,
    PRIMARY_CHOICE_NAME,
    SUBSET,
    PubMedQAExample,
    assert_state_keys_allowed,
    balanced_sample,
    build_state,
    download_labeled_jsonl,
    iter_labeled_rows,
    load_labeled_split,
    map_row,
    normalize_choice_label,
)

FIXTURE_JSONL = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pubmedqa"
    / "pqa_labeled_subset.jsonl"
)
FIXTURE_TEXT = FIXTURE_JSONL.read_text(encoding="utf-8")
VERSIONED_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "pubmedqa_answer_choice_schema_v1.json"
)


@pytest.mark.unit
@pytest.mark.parametrize("label", CHOICE_LABELS)
def test_normalize_choice_label_accepts_canonical(label: str) -> None:
    assert normalize_choice_label(label) == label
    assert normalize_choice_label(label.upper()) == label


@pytest.mark.unit
def test_normalize_choice_label_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="final_decision"):
        normalize_choice_label("unknown")


@pytest.mark.unit
def test_build_state_zips_contexts_in_order() -> None:
    state = build_state(
        "Is X true?",
        {"contexts": ["a", "b"], "labels": ["BACKGROUND", "RESULTS"]},
    )
    assert state == {
        "question": "Is X true?",
        "contexts": [
            {"label": "BACKGROUND", "text": "a"},
            {"label": "RESULTS", "text": "b"},
        ],
    }
    assert_state_keys_allowed(state)


@pytest.mark.unit
def test_build_state_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        build_state("Q?", {"contexts": ["a"], "labels": ["A", "B"]})


@pytest.mark.unit
def test_banned_state_keys_constant() -> None:
    assert "final_decision" in BANNED_STATE_KEYS
    assert "long_answer" in BANNED_STATE_KEYS


@pytest.mark.unit
def test_assert_state_keys_allowed_rejects_answer() -> None:
    with pytest.raises(ValueError, match="banned keys"):
        assert_state_keys_allowed({"question": "q", "answer": "yes"})


@pytest.mark.unit
def test_map_row_from_minimal_record() -> None:
    record = {
        "pubid": 42,
        "question": "Q?",
        "context": {"contexts": ["ctx"], "labels": ["BACKGROUND"]},
        "final_decision": "maybe",
    }
    ex = map_row(record)
    assert ex == PubMedQAExample(
        pubid=42,
        state={
            "question": "Q?",
            "contexts": [{"label": "BACKGROUND", "text": "ctx"}],
        },
        choice_label="maybe",
    )
    assert ex.subset == SUBSET


@pytest.mark.unit
def test_iter_labeled_rows_rejects_bad_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        list(iter_labeled_rows("{not json}\n"))


@pytest.mark.unit
def test_iter_labeled_rows_rejects_non_object() -> None:
    with pytest.raises(TypeError, match="JSON object"):
        list(iter_labeled_rows("[1]\n"))


@pytest.mark.unit
def test_load_fixture_subset_unbalanced() -> None:
    rows = load_labeled_split(jsonl_text=FIXTURE_TEXT)
    assert len(rows) == 12
    assert {r.choice_label for r in rows} == set(CHOICE_LABELS)


@pytest.mark.unit
def test_balanced_sample_from_fixture() -> None:
    all_rows = load_labeled_split(jsonl_text=FIXTURE_TEXT)
    sample = balanced_sample(all_rows, limit=12, seed=0)
    assert len(sample) == 12
    assert {r.choice_label for r in sample} == set(CHOICE_LABELS)
    counts = {label: 0 for label in CHOICE_LABELS}
    for row in sample:
        counts[row.choice_label] += 1
    assert counts == dict.fromkeys(CHOICE_LABELS, 4)


@pytest.mark.unit
def test_load_labeled_split_balanced_flag() -> None:
    rows = load_labeled_split(jsonl_text=FIXTURE_TEXT, balanced=True, limit=9, seed=1)
    assert len(rows) == 9
    assert {r.choice_label for r in rows} == set(CHOICE_LABELS)


@pytest.mark.unit
def test_load_labeled_split_limit_without_balance() -> None:
    rows = load_labeled_split(jsonl_text=FIXTURE_TEXT, limit=2)
    assert len(rows) == 2
    first = json.loads(FIXTURE_TEXT.splitlines()[0])
    assert rows[0].pubid == first["pubid"]


@pytest.mark.unit
def test_answer_choice_schema_compiles() -> None:
    decisions = compile_json_schema(ANSWER_CHOICE_SCHEMA)
    assert len(decisions) == 1
    assert decisions[0].name == PRIMARY_CHOICE_NAME
    assert decisions[0].syntax == "Choice"
    assert decisions[0].choices == CHOICE_LABELS


@pytest.mark.unit
def test_versioned_schema_matches_module() -> None:
    on_disk = json.loads(VERSIONED_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == ANSWER_CHOICE_SCHEMA


@pytest.mark.unit
def test_excluded_configs_documented() -> None:
    assert "pqa_artificial" in EXCLUDED_CONFIGS
    assert "pqa_unlabeled" in EXCLUDED_CONFIGS


@pytest.mark.unit
def test_download_labeled_jsonl_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/info"):
                payload = {
                    "dataset_info": {SUBSET: {"splits": {"train": {"num_examples": 1}}}}
                }
                return httpx.Response(200, json=payload)
            row = json.loads(FIXTURE_TEXT.splitlines()[0])
            payload = {"rows": [{"row": row}]}
            return httpx.Response(200, json=payload)

    real_client = httpx.Client

    def client_factory(*args: object, **kwargs: object) -> httpx.Client:
        transport = kwargs.get("transport", FakeTransport())
        assert isinstance(transport, httpx.BaseTransport)
        return real_client(*args, transport=transport)

    monkeypatch.setattr(httpx, "Client", client_factory)
    text = download_labeled_jsonl()
    assert len(list(iter_labeled_rows(text))) == 1


@pytest.mark.unit
def test_download_labeled_jsonl_uses_injected_client() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host != "datasets-server.huggingface.co":
                return httpx.Response(404)
            if request.url.path.endswith("/info"):
                payload = {
                    "dataset_info": {SUBSET: {"splits": {"train": {"num_examples": 1}}}}
                }
                return httpx.Response(200, json=payload)
            row = json.loads(FIXTURE_TEXT.splitlines()[0])
            return httpx.Response(200, json={"rows": [{"row": row}]})

    client = httpx.Client(transport=FakeTransport())
    text = download_labeled_jsonl(client=client)
    assert len(list(iter_labeled_rows(text))) == 1


@pytest.mark.unit
def test_load_labeled_split_downloads_when_jsonl_missing() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/info"):
                payload = {
                    "dataset_info": {SUBSET: {"splits": {"train": {"num_examples": 2}}}}
                }
                return httpx.Response(200, json=payload)
            lines = FIXTURE_TEXT.splitlines()[:2]
            rows = [{"row": json.loads(line)} for line in lines]
            return httpx.Response(200, json={"rows": rows})

    client = httpx.Client(transport=FakeTransport())
    rows = load_labeled_split(client=client, limit=2)
    assert len(rows) == 2


@pytest.mark.unit
def test_load_labeled_split_opens_ephemeral_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_download(*, client: httpx.Client | None = None) -> str:
        calls.append("download")
        assert client is None
        return FIXTURE_TEXT

    monkeypatch.setattr(
        "typevet.eval_pubmedqa.download_labeled_jsonl",
        fake_download,
    )
    rows = load_labeled_split(limit=1)
    assert calls == ["download"]
    assert len(rows) == 1
