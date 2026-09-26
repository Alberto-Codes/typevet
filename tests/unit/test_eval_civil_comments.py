"""Offline tests for Civil Comments loader and is_toxic Noul fixture (#72)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from typevet.domain import compile_json_schema
from typevet.evaluation.datasets.civil_comments import (
    DEFAULT_TIER_SEED,
    IS_TOXIC_NOUL_SCHEMA,
    IS_TOXIC_NOUL_SCHEMA_VERSION,
    PRIMARY_NOUL_NAME,
    TIER_A_LIMIT,
    TIER_B_LIMIT,
    TOXICITY_THRESHOLD,
    CivilCommentsExample,
    balanced_sample,
    download_test_csv,
    iter_test_rows,
    load_test_split,
    load_tier,
    load_tier_a,
    load_tier_b,
    map_row,
    proxy_label_for_toxicity,
    tier_limit,
    tier_manifest,
)

FIXTURE_CSV = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "civil_comments"
    / "test_subset.csv"
)
FIXTURE_TEXT = FIXTURE_CSV.read_text(encoding="utf-8")
VERSIONED_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "civil_comments_is_toxic_noul_schema_v1.json"
)
TIER_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "civil_comments_tier_manifest_v1.yaml"
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("score", "label"),
    [(0.49, "not_toxic"), (0.5, "toxic"), (0.9, "toxic")],
)
def test_proxy_label_at_tau_half(score: float, label: str) -> None:
    assert proxy_label_for_toxicity(score) == label
    assert TOXICITY_THRESHOLD == 0.5


@pytest.mark.unit
def test_map_row_sets_fields() -> None:
    ex = map_row("hello", 0.82)
    assert ex == CivilCommentsExample(text="hello", toxicity=0.82, label="toxic")


@pytest.mark.unit
def test_iter_test_rows_ignores_extra_columns() -> None:
    rows = list(iter_test_rows(FIXTURE_TEXT))
    assert len(rows) == 8
    assert rows[0][1] == pytest.approx(0.05)


@pytest.mark.unit
def test_iter_test_rows_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="toxicity"):
        list(iter_test_rows("text,label\nhello,0.1\n"))


@pytest.mark.unit
def test_iter_test_rows_rejects_empty_csv() -> None:
    with pytest.raises(ValueError, match="empty"):
        list(iter_test_rows(""))


@pytest.mark.unit
def test_load_fixture_subset_unbalanced() -> None:
    rows = load_test_split(csv_text=FIXTURE_TEXT)
    assert len(rows) == 8
    assert sum(1 for r in rows if r.label == "toxic") == 4


@pytest.mark.unit
def test_balanced_sample_from_fixture() -> None:
    all_rows = load_test_split(csv_text=FIXTURE_TEXT)
    sample = balanced_sample(all_rows, limit=4, seed=0)
    assert len(sample) == 4
    assert sum(1 for r in sample if r.label == "toxic") == 2
    assert sum(1 for r in sample if r.label == "not_toxic") == 2


@pytest.mark.unit
def test_load_test_split_balanced_flag() -> None:
    rows = load_test_split(csv_text=FIXTURE_TEXT, balanced=True, limit=4, seed=1)
    assert len(rows) == 4
    assert {r.label for r in rows} == {"toxic", "not_toxic"}


@pytest.mark.unit
def test_tier_limits_and_helpers() -> None:
    assert tier_limit("A") == TIER_A_LIMIT == 200
    assert tier_limit("B") == TIER_B_LIMIT == 2000
    rows_a = load_tier_a(csv_text=FIXTURE_TEXT, seed=0)
    assert len(rows_a) == 8
    assert sum(1 for r in rows_a if r.label == "toxic") == 4
    rows_b = load_tier_b(csv_text=FIXTURE_TEXT, seed=0)
    assert len(rows_b) == len(rows_a)
    assert load_tier("A", csv_text=FIXTURE_TEXT, seed=0) == rows_a


@pytest.mark.unit
def test_tier_manifest_matches_yaml() -> None:
    on_disk = yaml.safe_load(TIER_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert tier_manifest() == on_disk
    assert on_disk["default_seed"] == DEFAULT_TIER_SEED


@pytest.mark.unit
def test_is_toxic_noul_schema_compiles() -> None:
    decisions = compile_json_schema(IS_TOXIC_NOUL_SCHEMA)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.name == PRIMARY_NOUL_NAME
    assert decision.syntax == "Bool"
    assert decision.return_probabilities is True


@pytest.mark.unit
def test_versioned_json_fixture_matches_python_schema() -> None:
    on_disk = json.loads(VERSIONED_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk == IS_TOXIC_NOUL_SCHEMA
    assert IS_TOXIC_NOUL_SCHEMA_VERSION == "1"


@pytest.mark.unit
def test_download_test_csv_uses_injected_client() -> None:
    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host != "datasets-server.huggingface.co":
                return httpx.Response(404)
            if "/info" in str(request.url):
                body = {
                    "dataset_info": {
                        "default": {
                            "splits": {"test": {"num_examples": 2}},
                        }
                    }
                }
                return httpx.Response(200, json=body)
            return httpx.Response(
                200,
                json={
                    "rows": [
                        {"row": {"text": "a", "toxicity": 0.1}},
                        {"row": {"text": "b", "toxicity": 0.9}},
                    ]
                },
            )

    client = httpx.Client(transport=FakeTransport())
    csv_text = download_test_csv(client=client, page_size=2)
    rows = load_test_split(csv_text=csv_text)
    assert len(rows) == 2
    assert rows[1].label == "toxic"


@pytest.mark.unit
def test_load_test_split_downloads_when_csv_text_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(**kwargs: object) -> str:
        return FIXTURE_TEXT

    monkeypatch.setattr(
        "typevet.evaluation.datasets.civil_comments.download_test_csv",
        fake_download,
    )
    rows = load_test_split(limit=2)
    assert len(rows) == 2
