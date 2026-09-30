"""Live image pixel limits on the pinned Gemma 4 vision alias ([#204][i204]).

The test sends one ``Noul`` per synthetic PNG to the llama.cpp router through
``open_gemma_native_vision_judgment``. The rungs are squares of 256, 1024,
2048 and 4096 px, plus 64x4096 and 4096x64 (width x height). Each rung
records its outcome, the typed error class, the prompt tokens and the latency.

The property is a typed outcome within the time budget. A rung passes when it
returns a valid ``NoulAnswer`` or raises a ``GenerationError`` subclass. The
test does not check the answer value. Set ``TYPEVET_LIVE_RECEIPT_DIR`` to
write ``pixel_limits_llama_cpp_receipt.json``.

Examples:
    ```bash
    export TYPEVET_REQUIRE_LIVE=1
    export TYPEVET_LIVE_RECEIPT_DIR=tests/fixtures/runtime_limits
    uv run pytest -q -m live tests/live/test_image_pixel_limits_live.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: Factory

[i204]: https://github.com/Alberto-Codes/typevet/issues/204
"""

from __future__ import annotations

import json
import os
import struct
import time
import zlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.live.gate import gate_live
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)
from typevet.domain.errors import GenerationError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput
from typevet_evals.experiment_identity import read_baseline_commit

_LLAMA = load_llama_settings()
_ALIAS = "gemma-4-31b-kv9-q4km-mm"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNGS = ((256, 256), (1024, 1024), (2048, 2048), (4096, 4096), (64, 4096), (4096, 64))
# One HTTP read may wait this long. A rung also gets a wall budget, because
# one judge call makes several HTTP requests.
_HTTP_TIMEOUT = 600.0
_WALL_BUDGET = 900.0
_QUESTIONS = {"filled": Noul(instructions="Is the attached image filled with red?")}
_STATE = "Look at the attached image."
# Failures that are not typevet errors. Record them; do not stop the ladder.
_UNTYPED = (httpx.HTTPError, ValueError, KeyError, TypeError, OSError, RuntimeError)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _red_png(width: int, height: int) -> bytes:
    """Encode a ``width`` x ``height`` truecolour PNG filled with red.

    Returns:
        Complete PNG bytes.
    """
    scanline = b"\x00" + b"\xff\x00\x00" * width
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(scanline * height, 9))
        + _chunk(b"IEND", b"")
    )


@pytest.fixture
def vision_alias() -> str:
    """Require the router and the pinned vision alias in the catalog."""
    gate_live(replace(_LLAMA, default_model=_ALIAS))
    return _ALIAS


def _run_rung(width: int, height: int) -> dict[str, Any]:
    """Send one ``Noul`` with one synthetic PNG and classify the outcome.

    Returns:
        One receipt row for the rung.
    """
    png = _red_png(width, height)
    settings = replace(_LLAMA, multimodal_model=_ALIAS, timeout=_HTTP_TIMEOUT)
    row: dict[str, Any] = {
        "width": width,
        "height": height,
        "png_bytes": len(png),
        "outcome": None,
        "error_class": None,
        "error_message": None,
        "noul": None,
        "prompt_tokens": None,
    }
    started = time.perf_counter()
    try:
        with open_gemma_native_vision_judgment(settings=settings) as session:
            response = session.port.judge(
                _STATE,
                _QUESTIONS,
                _ALIAS,
                media=(ImageInput(data=png, mime_type="image/png"),),
            )
    except GenerationError as exc:
        row.update(outcome="typed_error", error_class=type(exc).__name__)
        row["error_message"] = str(exc)[:300]
    except _UNTYPED as exc:
        # An untyped failure is a finding: record it, then fail after all rungs.
        row.update(outcome="untyped_error", error_class=type(exc).__qualname__)
        row["error_message"] = str(exc)[:300]
    else:
        answer = response.answers["filled"]
        valid = isinstance(answer, NoulAnswer) and 0.0 <= answer.noul <= 1.0
        row["outcome"] = "answer" if valid else "invalid_answer"
        row["noul"] = answer.noul if isinstance(answer, NoulAnswer) else None
        row["prompt_tokens"] = response.usage.input_tokens
    row["latency_s"] = round(time.perf_counter() - started, 3)
    return row


def _router_identity(model: str) -> dict[str, Any]:
    base = _LLAMA.base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=60.0) as client:
        router = client.get("/props").raise_for_status().json()
        served = client.get("/props", params={"model": model}).raise_for_status()
    props = served.json()
    return {
        "router_build_info": router.get("build_info"),
        "model_build_info": props.get("build_info"),
        "model_alias": props.get("model_alias"),
        "modalities": props.get("modalities"),
        "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
    }


def _write_receipt(receipt: dict[str, Any]) -> None:
    out_dir = os.environ.get("TYPEVET_LIVE_RECEIPT_DIR")
    if not out_dir:
        return
    path = Path(out_dir) / "pixel_limits_llama_cpp_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


@pytest.mark.live
def test_image_pixel_rungs_give_typed_outcome_within_budget(
    vision_alias: str,
) -> None:
    """Each pixel rung returns a valid answer or a typed error within budget."""
    rows = [_run_rung(width, height) for width, height in _RUNGS]
    receipt = {
        "issue": 204,
        "alias": vision_alias,
        "typevet_commit": read_baseline_commit(_REPO_ROOT),
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "http_timeout_s": _HTTP_TIMEOUT,
        "wall_budget_s": _WALL_BUDGET,
        "router": _router_identity(vision_alias),
        "rungs": rows,
    }
    _write_receipt(receipt)
    for row in rows:
        rung = f"{row['width']}x{row['height']}"
        assert row["outcome"] in {"answer", "typed_error"}, (rung, row)
        assert row["latency_s"] <= _WALL_BUDGET, (rung, row)
