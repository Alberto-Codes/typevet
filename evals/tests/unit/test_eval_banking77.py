"""Offline tests for Banking77 test-split loader (#58)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from typevet.domain import compile_json_schema
from typevet_evals.datasets.banking77 import (
    FRAUD_INTENTS,
    PRIMARY_NOUL_NAME,
    REPORTS_UNAUTHORIZED_NOUL_SCHEMA,
    Banking77Example,
    balanced_sample,
    download_test_csv,
    iter_test_rows,
    load_test_split,
    map_row,
    proxy_label_for_intent,
)

FIXTURE_CSV = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "banking77"
    / "test_subset.csv"
)
FIXTURE_TEXT = FIXTURE_CSV.read_text(encoding="utf-8")


@pytest.mark.unit
@pytest.mark.parametrize("intent", FRAUD_INTENTS)
def test_fraud_intents_map_to_fraud(intent: str) -> None:
    assert proxy_label_for_intent(intent) == "fraud"


@pytest.mark.unit
def test_non_fraud_intent_maps_to_not_fraud() -> None:
    assert proxy_label_for_intent("card_arrival") == "not_fraud"


@pytest.mark.unit
def test_map_row_sets_fields() -> None:
    ex = map_row("charge?", "card_payment_not_recognised")
    assert ex == Banking77Example(
        text="charge?",
        intent="card_payment_not_recognised",
        proxy_label="fraud",
    )


@pytest.mark.unit
def test_iter_test_rows_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="category"):
        list(iter_test_rows("text,label\nhello,world\n"))


@pytest.mark.unit
def test_iter_test_rows_rejects_empty_csv() -> None:
    with pytest.raises(ValueError, match="empty"):
        list(iter_test_rows(""))


@pytest.mark.unit
def test_load_fixture_subset_unbalanced() -> None:
    rows = load_test_split(csv_text=FIXTURE_TEXT)
    assert len(rows) == 8
    assert sum(1 for r in rows if r.proxy_label == "fraud") == 6


@pytest.mark.unit
def test_balanced_sample_from_fixture() -> None:
    all_rows = load_test_split(csv_text=FIXTURE_TEXT)
    sample = balanced_sample(all_rows, limit=4, seed=0)
    assert len(sample) == 4
    assert sum(1 for r in sample if r.proxy_label == "fraud") == 2
    assert sum(1 for r in sample if r.proxy_label == "not_fraud") == 2


@pytest.mark.unit
def test_load_test_split_balanced_flag() -> None:
    rows = load_test_split(csv_text=FIXTURE_TEXT, balanced=True, limit=4, seed=1)
    assert len(rows) == 4
    labels = {r.proxy_label for r in rows}
    assert labels == {"fraud", "not_fraud"}


@pytest.mark.unit
def test_load_test_split_limit_without_balance() -> None:
    rows = load_test_split(csv_text=FIXTURE_TEXT, limit=3)
    assert len(rows) == 3
    assert rows[0].intent == "card_arrival"


@pytest.mark.unit
def test_reports_unauthorized_noul_schema_compiles() -> None:
    decisions = compile_json_schema(REPORTS_UNAUTHORIZED_NOUL_SCHEMA)
    assert len(decisions) == 1
    assert decisions[0].name == PRIMARY_NOUL_NAME
    assert decisions[0].syntax == "Bool"


@pytest.mark.unit
def test_download_test_csv_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=FIXTURE_TEXT)

    real_client = httpx.Client

    def client_factory(*args: object, **kwargs: object) -> httpx.Client:
        transport = kwargs.get("transport", FakeTransport())
        assert isinstance(transport, httpx.BaseTransport)
        return real_client(*args, transport=transport)

    monkeypatch.setattr(httpx, "Client", client_factory)
    assert download_test_csv() == FIXTURE_TEXT


@pytest.mark.unit
def test_download_test_csv_uses_injected_client() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            assert request.url.host == "raw.githubusercontent.com"
            return httpx.Response(200, text=FIXTURE_TEXT)

    client = httpx.Client(transport=FakeTransport())
    assert download_test_csv(client=client) == FIXTURE_TEXT


@pytest.mark.unit
def test_load_test_split_downloads_when_csv_text_missing() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=FIXTURE_TEXT)

    client = httpx.Client(transport=FakeTransport())
    rows = load_test_split(client=client, limit=2)
    assert len(rows) == 2


@pytest.mark.unit
def test_load_test_split_opens_ephemeral_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_download(url: str = "", client: httpx.Client | None = None) -> str:
        calls.append("download")
        assert client is None
        return FIXTURE_TEXT

    monkeypatch.setattr(
        "typevet_evals.datasets.banking77.download_test_csv",
        fake_download,
    )
    rows = load_test_split(limit=1)
    assert calls == ["download"]
    assert len(rows) == 1
