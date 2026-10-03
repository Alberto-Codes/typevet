r"""Opt-in live doodle Choice run with a key-free receipt (#412).

Runs the ``doodle_duel`` command once against a served model: 5 doodles from
each of the 24 categories, one ``Choice`` per doodle. The test asserts that
the run did not stop, the row count and the full distribution keys. It
prints the accuracy and applies no accuracy threshold.

The test skips unless ``TYPEVET_DOODLE_RECEIPT`` names the receipt file. It
fails when ``TYPEVET_REQUIRE_LIVE`` is truthy and that variable is missing.
The receipt path must not exist. ``TYPEVET_BACKEND`` selects the backend, as
in ``open_judgment``; ``fake`` is refused. A cache miss streams the first
recognized drawings of each category from Google's public bucket into
``TYPEVET_QUICKDRAW_CACHE``.

Examples:
    ```bash
    TYPEVET_DOODLE_RECEIPT=evals/fixtures/quickdraw/receipts/doodle_llama_cpp.json \
      uv run pytest evals/tests/live/test_doodle_duel_live.py -m live -q -s
    ```

See Also:
    - [typevet_evals.cli.doodle_duel][]: the command this test runs
    - [typevet_evals.doodle_duel.runner][]: run, metrics and receipt
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from typevet_evals.cli import doodle_duel
from typevet_evals.doodle_duel import DOODLE_CATEGORIES
from typevet_evals.runner.live_gate import require_live_enabled

pytestmark = pytest.mark.live

_RECEIPT_ENV = "TYPEVET_DOODLE_RECEIPT"
_PER_CATEGORY = 5


def _receipt_path() -> Path:
    raw = os.environ.get(_RECEIPT_ENV, "").strip()
    if not raw:
        reason = f"{_RECEIPT_ENV} not set"
        if require_live_enabled():
            pytest.fail(reason)
        pytest.skip(reason)
    return Path(raw)


def test_doodle_duel_live_receipt() -> None:
    """Judge 120 doodles once and check the receipt shape."""
    path = _receipt_path()
    code = doodle_duel.main(
        ["--receipt", str(path), "--per-category", str(_PER_CATEGORY), "--require-live"]
    )
    assert code == 0

    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["stopped"] is None, receipt["stopped"]
    rows = receipt["rows"]
    assert len(rows) == _PER_CATEGORY * len(DOODLE_CATEGORIES)
    for row in rows:
        assert list(row["probabilities"]) == list(DOODLE_CATEGORIES)
    print(f"accuracy {receipt['metrics']['accuracy']}")
