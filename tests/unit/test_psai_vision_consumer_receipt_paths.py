"""Unit tests: consumer receipt path helpers ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_receipt_paths.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_receipt][]: receipt helpers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from typevet.domain.judgment_answers import Answer, NoulAnswer
from typevet.evaluation.psai_vision_consumer_receipt import (
    consumer_receipt_basename,
    resolve_receipt_write_path,
    serialize_answer,
    write_receipt_json,
)


@pytest.mark.unit
def test_serialize_answer_rejects_unknown_type() -> None:
    """Unknown answer types raise ``TypeError``."""
    bad_answer = cast(Answer, object())
    with pytest.raises(TypeError):
        serialize_answer(bad_answer)


@pytest.mark.unit
def test_serialize_noul_answer_roundtrip() -> None:
    """Noul answers serialize to JSON-friendly dicts."""
    payload = serialize_answer(NoulAnswer(noul=0.5))
    assert payload == {"kind": "Noul", "noul": 0.5}


@pytest.mark.unit
def test_resolve_receipt_write_path_appends_attempt_suffix(tmp_path: Path) -> None:
    """Second write target uses ``-attempt-N`` when base exists."""
    first = resolve_receipt_write_path(
        tmp_path,
        protocol_revision=2,
        wheel_sha256="abc",
        model_id="model",
    )
    first.write_text("{}", encoding="utf-8")
    second = resolve_receipt_write_path(
        tmp_path,
        protocol_revision=2,
        wheel_sha256="abc",
        model_id="model",
    )
    assert second.name.endswith("-attempt-1.json")


@pytest.mark.unit
def test_write_receipt_json_and_basename(tmp_path: Path) -> None:
    """Basename helper stays stable; write helper emits JSON."""
    name = consumer_receipt_basename(
        protocol_revision=2,
        wheel_sha256="deadbeef",
        model_id="gemma-4",
    )
    assert name.startswith("consumer-receipt-p2-")
    path = tmp_path / name
    write_receipt_json(path, {"ok": True})
    assert json.loads(path.read_text(encoding="utf-8"))["ok"] is True
