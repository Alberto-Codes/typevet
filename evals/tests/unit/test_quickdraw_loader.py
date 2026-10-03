"""Unit tests for the Quick, Draw! first-N loader and its cache (#412)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from typevet_evals.datasets.quickdraw import (
    CACHE_ENV_VAR,
    STROKES_URL,
    Doodle,
    fetch_doodles,
    first_recognized,
    parse_line,
    resolve_cache_dir,
)

pytestmark = pytest.mark.unit


def _line(key_id: str, *, word: str = "cat", recognized: bool = True) -> str:
    return json.dumps(
        {
            "word": word,
            "countrycode": "US",
            "timestamp": "2017-03-01 00:00:00.00000 UTC",
            "recognized": recognized,
            "key_id": key_id,
            "drawing": [[[0, 10, 20], [5, 15, 25]], [[255], [0]]],
        }
    )


def _refuse(request: httpx.Request) -> httpx.Response:
    msg = f"unexpected request {request.url}"
    raise AssertionError(msg)


def test_parse_line_reads_word_key_id_and_strokes() -> None:
    doodle = parse_line(_line("123"))
    assert doodle == Doodle(
        key_id="123",
        word="cat",
        countrycode="US",
        strokes=(((0, 5), (10, 15), (20, 25)), ((255, 0),)),
    )


def test_first_recognized_skips_unrecognized_and_stops_at_n() -> None:
    read: list[str] = []

    def lines() -> Iterator[str]:
        for key in ("u1", "r1", "u2", "r2", "r3"):
            read.append(key)
            yield _line(key, recognized=key.startswith("r"))
        msg = "read past the end"
        raise AssertionError(msg)

    kept = first_recognized(lines(), 2)
    assert [json.loads(line)["key_id"] for line in kept] == ["r1", "r2"]
    assert read == ["u1", "r1", "u2", "r2"]


def test_resolve_cache_dir_prefers_argument_then_env_then_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env"))
    assert resolve_cache_dir(tmp_path / "arg") == tmp_path / "arg"
    assert resolve_cache_dir() == tmp_path / "env"
    monkeypatch.delenv(CACHE_ENV_VAR)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    expected = tmp_path / "home" / ".cache" / "typevet" / "quickdraw"
    assert resolve_cache_dir() == expected


def test_fetch_reads_cache_without_request(tmp_path: Path) -> None:
    cached = tmp_path / "cat.first2.ndjson"
    cached.write_text(f"{_line('a')}\n{_line('b')}\n", encoding="utf-8")
    with httpx.Client(transport=httpx.MockTransport(_refuse)) as client:
        doodles = fetch_doodles("cat", 2, cache_dir=tmp_path, client=client)
    assert [d.key_id for d in doodles] == ["a", "b"]


def test_fetch_writes_first_n_recognized_to_cache(tmp_path: Path) -> None:
    word = "hot dog"
    body = "\n".join(
        _line(key, word=word, recognized=key.startswith("r"))
        for key in ("u1", "r1", "r2", "u2", "r3")
    )
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text=body + "\n")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        doodles = fetch_doodles(word, 2, cache_dir=tmp_path, client=client)

    assert seen == [STROKES_URL.format(word="hot%20dog")]
    assert [d.key_id for d in doodles] == ["r1", "r2"]
    cached = (tmp_path / "hot dog.first2.ndjson").read_text(encoding="utf-8")
    assert [json.loads(x)["key_id"] for x in cached.splitlines()] == ["r1", "r2"]
    with httpx.Client(transport=httpx.MockTransport(_refuse)) as client:
        again = fetch_doodles(word, 2, cache_dir=tmp_path, client=client)
    assert again == doodles
