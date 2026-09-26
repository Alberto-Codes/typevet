"""Offline tests for DIFrauD loader and is_scam Noul fixture (#59)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from typevet.domain import compile_json_schema
from typevet.eval_difraud import (
    DEFAULT_DOMAIN,
    IS_SCAM_NOUL_SCHEMA,
    IS_SCAM_NOUL_SCHEMA_VERSION,
    PRIMARY_NOUL_NAME,
    SUPPORTED_DOMAINS,
    DIFrauDExample,
    iter_test_rows,
    load_test_split,
    map_row,
)

FIXTURE_JSONL = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "difraud"
    / "sms_test_subset.jsonl"
)
FIXTURE_TEXT = FIXTURE_JSONL.read_text(encoding="utf-8")
VERSIONED_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "difraud_is_scam_noul_schema_v1.json"
)


@pytest.mark.unit
@pytest.mark.parametrize(("flag", "label"), [(1, "scam"), (True, "scam"), (0, "legit")])
def test_map_row(flag: int | bool, label: str) -> None:
    ex = map_row("You won a prize", flag)
    assert ex == DIFrauDExample(
        text="You won a prize", label=label, domain=DEFAULT_DOMAIN
    )


@pytest.mark.unit
def test_default_domain_is_sms() -> None:
    assert DEFAULT_DOMAIN == "sms"


@pytest.mark.unit
def test_unsupported_domain_raises() -> None:
    with pytest.raises(ValueError, match="unsupported DIFrauD domain"):
        load_test_split(domain="email", jsonl_text=FIXTURE_TEXT)


@pytest.mark.unit
@pytest.mark.parametrize("domain", SUPPORTED_DOMAINS)
def test_supported_domains_accept_fixture(domain: str) -> None:
    rows = load_test_split(jsonl_text=FIXTURE_TEXT, domain=domain, limit=2, seed=0)
    assert len(rows) == 2
    assert rows[0].domain == domain


@pytest.mark.unit
def test_iter_test_rows_rejects_bad_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        list(iter_test_rows("{not json}\n"))


@pytest.mark.unit
def test_iter_test_rows_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="text"):
        list(iter_test_rows('{"label":0}\n'))


@pytest.mark.unit
def test_load_fixture_subset_preserves_class_mix() -> None:
    rows = load_test_split(jsonl_text=FIXTURE_TEXT, limit=4, seed=0)
    assert len(rows) == 4
    assert sum(1 for r in rows if r.label == "scam") == 2
    assert sum(1 for r in rows if r.label == "legit") == 2


@pytest.mark.unit
def test_load_respects_limit_after_shuffle() -> None:
    rows = load_test_split(jsonl_text=FIXTURE_TEXT, limit=2, seed=99)
    assert len(rows) == 2


@pytest.mark.unit
def test_is_scam_noul_schema_compiles() -> None:
    decisions = compile_json_schema(IS_SCAM_NOUL_SCHEMA)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.name == PRIMARY_NOUL_NAME
    assert decision.syntax == "Bool"
    assert decision.return_probabilities is True
    assert "scam" in decision.question.lower()


@pytest.mark.unit
def test_is_scam_schema_has_no_banking_fraud_fields() -> None:
    props = IS_SCAM_NOUL_SCHEMA["properties"]
    assert "reports_unauthorized" not in props
    assert "fraud_type" not in props


@pytest.mark.unit
def test_versioned_json_fixture_matches_python_schema() -> None:
    on_disk = json.loads(VERSIONED_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == IS_SCAM_NOUL_SCHEMA
    assert IS_SCAM_NOUL_SCHEMA_VERSION == "1"


@pytest.mark.unit
def test_download_test_jsonl_uses_injected_client() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            assert "difraud/difraud" in str(request.url)
            assert request.url.path.endswith("/sms/test.jsonl")
            return httpx.Response(200, text=FIXTURE_TEXT)

    client = httpx.Client(transport=FakeTransport())
    rows = load_test_split(client=client, limit=2, seed=0)
    assert len(rows) == 2


@pytest.mark.unit
def test_download_phishing_domain_path() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            assert "/phishing/test.jsonl" in str(request.url)
            return httpx.Response(200, text=FIXTURE_TEXT)

    client = httpx.Client(transport=FakeTransport())
    rows = load_test_split(domain="phishing", client=client, limit=1)
    assert rows[0].domain == "phishing"
