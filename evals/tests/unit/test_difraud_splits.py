"""Offline tests for the DIFrauD SMS train, validation and held-out splits (#307).

The JSONL here is synthetic: no upstream text enters the repository.
"""

from __future__ import annotations

import json

import httpx
import pytest

from typevet_evals.datasets.difraud import (
    PINNED_REVISION,
    PRIOR_MEASURED_LIMIT,
    PRIOR_MEASURED_SEED,
    DIFrauDSplits,
    build_splits,
    load_splits,
    load_test_split,
    prior_measured_ids,
    record_id,
)
from typevet_evals.throughput.public_workload import DIFRAUD_LIMIT, SEED

pytestmark = pytest.mark.unit

SHARED_TRAIN_TEST = "shared train and test message"
SHARED_VALIDATION_TEST = "shared validation and test message"
SHARED_TRAIN_VALIDATION = "shared train and validation message"


def _jsonl(texts: list[str]) -> str:
    return "\n".join(
        json.dumps({"text": text, "label": index % 2})
        for index, text in enumerate(texts)
    )


def _test_texts() -> list[str]:
    texts = [f"test message {index}" for index in range(540)]
    texts[3] = SHARED_TRAIN_TEST
    texts[7] = SHARED_VALIDATION_TEST
    return texts


TRAIN_JSONL = _jsonl(
    [f"train message {index}" for index in range(40)]
    + [SHARED_TRAIN_TEST, SHARED_TRAIN_VALIDATION, "train message 0"]
)
VALIDATION_JSONL = _jsonl(
    [f"validation message {index}" for index in range(20)]
    + [SHARED_VALIDATION_TEST, SHARED_TRAIN_VALIDATION]
)
TEST_JSONL = _jsonl(_test_texts())


def _splits(seed: int = 0) -> DIFrauDSplits:
    return build_splits(
        train_jsonl=TRAIN_JSONL,
        validation_jsonl=VALIDATION_JSONL,
        test_jsonl=TEST_JSONL,
        seed=seed,
    )


def _ids(splits: DIFrauDSplits) -> dict[str, list[str]]:
    return {
        "train": [row.record_id for row in splits.train],
        "validation": [row.record_id for row in splits.validation],
        "held_out": [row.record_id for row in splits.held_out],
    }


def test_no_record_id_appears_in_two_splits() -> None:
    ids = _ids(_splits())
    for name, values in ids.items():
        assert len(values) == len(set(values)), name
    assert not set(ids["train"]) & set(ids["validation"])
    assert not set(ids["train"]) & set(ids["held_out"])
    assert not set(ids["validation"]) & set(ids["held_out"])
    assert record_id(SHARED_TRAIN_VALIDATION) in ids["train"]
    assert record_id(SHARED_TRAIN_VALIDATION) not in ids["validation"]


def test_held_out_excludes_every_prior_measured_record() -> None:
    prior_rows = load_test_split(jsonl_text=TEST_JSONL, limit=DIFRAUD_LIMIT, seed=SEED)
    expected = {record_id(row.text) for row in prior_rows}
    splits = _splits()
    held_out = set(_ids(splits)["held_out"])

    assert len(expected) == DIFRAUD_LIMIT
    assert splits.prior_measured_ids == expected
    assert prior_measured_ids(TEST_JSONL) == expected
    assert not held_out & expected
    test_ids = {record_id(text) for text in _test_texts()}
    cross_split = {record_id(SHARED_TRAIN_TEST), record_id(SHARED_VALIDATION_TEST)}
    assert held_out == test_ids - expected - cross_split
    assert held_out


def test_prior_selection_matches_the_236_workload_constants() -> None:
    assert (PRIOR_MEASURED_LIMIT, PRIOR_MEASURED_SEED) == (DIFRAUD_LIMIT, SEED)


def test_prior_selection_ignores_the_slice_seed() -> None:
    assert _splits(seed=0).prior_measured_ids == _splits(seed=5).prior_measured_ids


def test_same_seed_gives_the_same_slices() -> None:
    assert _splits(seed=3) == _splits(seed=3)
    assert _ids(_splits(seed=3)) != _ids(_splits(seed=4))
    for name in ("train", "validation", "held_out"):
        assert sorted(_ids(_splits(seed=3))[name]) == sorted(
            _ids(_splits(seed=4))[name]
        )


def test_rows_carry_their_split_name() -> None:
    splits = _splits()
    assert {row.example.split for row in splits.train} == {"train"}
    assert {row.example.split for row in splits.validation} == {"validation"}
    assert {row.example.split for row in splits.held_out} == {"test"}
    assert {row.example.label for row in splits.train} == {"scam", "legit"}


def test_record_id_is_a_stable_text_hash() -> None:
    assert record_id("hello") == record_id("hello")
    assert record_id("hello") != record_id("hello ")
    assert record_id("hello").startswith("sha256:")


def test_load_splits_fetches_each_file_at_the_pinned_revision() -> None:
    bodies = {
        "train.jsonl": TRAIN_JSONL,
        "validation.jsonl": VALIDATION_JSONL,
        "test.jsonl": TEST_JSONL,
    }
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text=bodies[request.url.path.rsplit("/", 1)[1]])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        loaded = load_splits(client=client)

    assert loaded == _splits()
    assert len(seen) == 3
    for url in seen:
        assert f"/resolve/{PINNED_REVISION}/sms/" in url


def test_load_splits_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        load_splits(client=client)
