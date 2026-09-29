"""Offline tests for CLINC150 plus domain-sharded loader (#73)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from typevet.domain import compile_json_schema
from typevet_evals.datasets.clinc import (
    BANNED_STATE_KEYS,
    DEFAULT_DOMAIN,
    IN_SCOPE_NOUL_SCHEMA,
    NOUL_LABELS,
    OOS_INTENT,
    PRIMARY_CHOICE_NAME,
    PRIMARY_NOUL_NAME,
    assert_state_keys_allowed,
    balanced_sample,
    build_state,
    choice_labels_for_domain,
    choice_schema_for_domain,
    domain_intent_map,
    domain_keys,
    intent_slug_from_record,
    load_plus_split,
    map_row,
    plus_intent_names,
)
from typevet_evals.datasets.clinc_download import download_plus_jsonl

FIXTURE_JSONL = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "clinc"
    / "plus_banking_micro.jsonl"
)
FIXTURE_TEXT = FIXTURE_JSONL.read_text(encoding="utf-8")
BANKING_CHOICE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals"
    / "fixtures"
    / "clinc_banking_intent_choice_schema_v1.json"
)
NOUL_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals"
    / "fixtures"
    / "clinc_in_scope_noul_schema_v1.json"
)
EVAL_CLINC_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "evals"
    / "src"
    / "typevet_evals"
    / "datasets"
    / "clinc.py"
)


@pytest.mark.unit
def test_module_loc_cap() -> None:
    assert len(EVAL_CLINC_SOURCE.read_text(encoding="utf-8").splitlines()) <= 300


@pytest.mark.unit
def test_no_banking77_import_in_loader() -> None:
    text = EVAL_CLINC_SOURCE.read_text(encoding="utf-8")
    assert "eval_banking77" not in text
    assert "FRAUD_INTENTS" not in text


@pytest.mark.unit
def test_domain_shard_geometry() -> None:
    assert len(domain_keys()) == 10
    for domain in domain_keys():
        assert len(domain_intent_map()[domain]) == 15
    assert len(choice_labels_for_domain(DEFAULT_DOMAIN)) == 15


@pytest.mark.unit
def test_default_domain_is_banking() -> None:
    assert DEFAULT_DOMAIN == "banking"


@pytest.mark.unit
def test_build_state_and_banned_keys() -> None:
    state = build_state("hello")
    assert state == {"text": "hello"}
    assert "intent" in BANNED_STATE_KEYS
    with pytest.raises(ValueError, match="banned keys"):
        assert_state_keys_allowed({"text": "x", "intent": "balance"})


@pytest.mark.unit
def test_map_row_filters_cross_domain() -> None:
    row = {"text": "card", "intent": "credit_limit"}
    assert map_row(row, domain=DEFAULT_DOMAIN) is None


@pytest.mark.unit
def test_map_row_report_fraud_is_choice_not_proxy() -> None:
    row = {"text": "fraud on account", "intent": "report_fraud"}
    ex = map_row(row, domain=DEFAULT_DOMAIN)
    assert ex is not None
    assert ex.choice_label == "report_fraud"
    assert ex.noul_label is None


@pytest.mark.unit
def test_include_oos_noul_track() -> None:
    in_row = {"text": "balance?", "intent": "balance"}
    oos_row = {"text": "random", "intent": OOS_INTENT}
    in_ex = map_row(in_row, domain=DEFAULT_DOMAIN, include_oos=True)
    oos_ex = map_row(oos_row, domain=DEFAULT_DOMAIN, include_oos=True)
    assert (
        in_ex is not None
        and in_ex.noul_label == "yes"
        and in_ex.choice_label == "balance"
    )
    assert (
        oos_ex is not None and oos_ex.noul_label == "no" and oos_ex.choice_label is None
    )


@pytest.mark.unit
def test_intent_index_resolution() -> None:
    names = ("balance", "transfer")
    assert intent_slug_from_record({"intent": 1}, names) == "transfer"


@pytest.mark.unit
def test_load_plus_split_from_fixture() -> None:
    rows = load_plus_split(domain=DEFAULT_DOMAIN, jsonl_text=FIXTURE_TEXT)
    assert len(rows) == 15
    assert {r.choice_label for r in rows} == set(
        choice_labels_for_domain(DEFAULT_DOMAIN)
    )


@pytest.mark.unit
def test_load_plus_split_balanced_limit() -> None:
    rows = load_plus_split(
        domain=DEFAULT_DOMAIN,
        jsonl_text=FIXTURE_TEXT,
        balanced=True,
        limit=15,
        seed=0,
    )
    assert len(rows) == 15


@pytest.mark.unit
def test_load_plus_split_with_oos() -> None:
    rows = load_plus_split(
        domain=DEFAULT_DOMAIN,
        jsonl_text=FIXTURE_TEXT,
        include_oos=True,
    )
    assert len(rows) == 16
    assert sum(1 for r in rows if r.noul_label == "no") == 1


@pytest.mark.unit
def test_banking_choice_schema_compiles_and_matches_fixture() -> None:
    schema = choice_schema_for_domain(DEFAULT_DOMAIN)
    on_disk = json.loads(BANKING_CHOICE_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema == on_disk
    decisions = compile_json_schema(schema)
    assert decisions[0].name == PRIMARY_CHOICE_NAME
    assert decisions[0].syntax == "Choice"
    assert len(decisions[0].choices) == 15


@pytest.mark.unit
def test_in_scope_noul_schema_compiles() -> None:
    on_disk = json.loads(NOUL_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == IN_SCOPE_NOUL_SCHEMA
    decisions = compile_json_schema(IN_SCOPE_NOUL_SCHEMA)
    assert decisions[0].name == PRIMARY_NOUL_NAME
    assert decisions[0].syntax == "Choice"
    assert decisions[0].choices == NOUL_LABELS
    assert decisions[0].return_probabilities is True


@pytest.mark.unit
def test_balanced_sample_preserves_oos() -> None:
    examples = load_plus_split(
        domain=DEFAULT_DOMAIN,
        jsonl_text=FIXTURE_TEXT,
        include_oos=True,
    )
    sampled = balanced_sample(examples, DEFAULT_DOMAIN, limit=15, seed=1)
    assert any(ex.noul_label == "no" for ex in sampled)


@pytest.mark.unit
def test_download_plus_jsonl_mocked() -> None:
    balance_index = plus_intent_names().index("balance")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/info"):
            body = {
                "dataset_info": {"plus": {"splits": {"train": {"num_examples": 1}}}}
            }
            return httpx.Response(200, json=body)
        return httpx.Response(
            200,
            json={
                "rows": [
                    {
                        "row": {
                            "text": "what is my balance",
                            "intent": balance_index,
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        jsonl = download_plus_jsonl(client=client, page_size=1)
    row = json.loads(jsonl.splitlines()[0])
    assert row["intent"] == "balance"
