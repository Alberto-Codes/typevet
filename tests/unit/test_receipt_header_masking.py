"""Unit tests: no gateway header value reaches a vLLM acceptance receipt (#348).

``run_acceptance`` records server bodies and error text in the receipt. A
gateway can echo a request header into either, so ``write_receipt`` masks
each ``TYPEVET_VLLM__HEADERS`` value as a whole token, raw and JSON-escaped,
the same way as the adapter errors. The key keeps its substring masking.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from typevet.adapters.inbound.backend_settings import VllmSettings
from typevet_evals.vllm_acceptance.core import write_receipt

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_KEY = "sk-KEY-SENTINEL-348"
_TENANT = "tenant-SENTINEL-348"
_QUOTED = 'route "SENTINEL" 348'


def _settings() -> VllmSettings:
    return VllmSettings(
        base_url="https://gw.example.com/vllm",
        model="m",
        api_key=_KEY,
        headers={"X-Tenant": _TENANT, "X-Route": _QUOTED, "X-Shard": "1"},
    )


def test_receipt_never_holds_a_header_value(tmp_path: Path) -> None:
    receipt = {
        "pins": {
            "version": {"status": 200, "body": {"echo": f"X-Tenant: {_TENANT}"}},
            "served_models": [{"id": f"echo {_TENANT}", "root": None}],
            "models_status": 1,
        },
        "error": {"message": f"route={_QUOTED}; HTTP 401: shard 1; key x{_KEY}y"},
        "calls": {"chat": 1},
    }
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt, settings=_settings())
    text = path.read_text(encoding="utf-8")
    written = json.loads(text)
    assert "SENTINEL" not in text
    assert written["error"]["message"] == "route=***; HTTP 401: shard ***; key x***y"
    assert written["pins"]["served_models"] == [{"id": "echo ***", "root": None}]
    assert written["pins"]["models_status"] == 1
    assert written["calls"] == {"chat": 1}


_MODEL = "gemma"
_STRUCTURAL = ("cases", "passed", "stopped", _MODEL)


def _receipt(echo: str) -> dict[str, object]:
    return {
        "stopped": None,
        "passed": False,
        "cases": [{"case": "cases"}],
        "combined": {"passed": 1},
        "pins": {
            "configured_model": _MODEL,
            "version": {"status": 200, "body": {"echo": f"saw {echo}"}},
        },
        "error": {"type": "RuntimeError", "message": f"gateway said {echo}"},
    }


@pytest.mark.parametrize("value", _STRUCTURAL)
def test_header_value_never_rewrites_keys_or_typevet_values(
    tmp_path: Path, value: str
) -> None:
    """Only echoed gateway text is masked; keys and typevet values stay (#348)."""
    settings = VllmSettings(
        base_url="https://gw.example.com/vllm",
        model=_MODEL,
        headers={"X-Tenant": value},
    )
    receipt = _receipt(value)
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt, settings=settings)
    written = json.loads(path.read_text(encoding="utf-8"))
    expected = _receipt("***")
    assert written == expected
    assert list(written) == list(receipt)
    assert written["pins"]["configured_model"] == _MODEL


def test_two_header_values_never_collapse_keys(tmp_path: Path) -> None:
    settings = VllmSettings(
        base_url="https://gw.example.com/vllm",
        model=_MODEL,
        headers={"X-A": "passed", "X-B": "stopped"},
    )
    receipt = _receipt("passed and stopped")
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt, settings=settings)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert set(written) == set(receipt)
    assert written["passed"] is False
    assert written["stopped"] is None
    assert written["error"]["message"] == "gateway said *** and ***"


def _key_receipt(body: object, root: object = None) -> dict[str, object]:
    return {
        "pins": {
            "version": {"status": 200, "body": body},
            "served_models": [{"id": "m", "root": root}],
        },
    }


def test_header_value_used_as_a_key_is_masked(tmp_path: Path) -> None:
    """A gateway can echo a header value as a JSON key in a body (#351)."""
    body = {_TENANT: "k", "nested": [{_TENANT: 1}]}
    receipt = _key_receipt(body, root={f"x {_TENANT}": 2})
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt, settings=_settings())
    text = path.read_text(encoding="utf-8")
    written = json.loads(text)
    assert "SENTINEL" not in text
    version = written["pins"]["version"]
    assert list(version) == ["status", "body"]
    assert version["body"] == {"***": "k", "nested": [{"***": 1}]}
    assert written["pins"]["served_models"] == [{"id": "m", "root": {"x ***": 2}}]


def test_masked_keys_never_collapse(tmp_path: Path) -> None:
    """Keys that mask to the same text get a numbered suffix, in order (#351)."""
    body = {_TENANT: 1, "***": 2, _QUOTED: 3, "***_2": 4, "shard 1": 5}
    path = tmp_path / "receipt.json"
    write_receipt(path, _key_receipt(body), settings=_settings())
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["pins"]["version"]["body"] == {
        "***_3": 1,
        "***": 2,
        "***_4": 3,
        "***_2": 4,
        "shard ***": 5,
    }


def test_header_value_equal_to_a_typevet_key_keeps_it(tmp_path: Path) -> None:
    """``status``, ``body``, ``id`` and ``root`` are typevet keys (#351)."""
    settings = VllmSettings(
        base_url="https://gw.example.com/vllm",
        model=_MODEL,
        headers={"X-A": "status", "X-B": "id", "X-C": "body", "X-D": "root"},
    )
    receipt = _key_receipt({"status": "ok"}, root={"id": "r"})
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt, settings=settings)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["pins"] == {
        "version": {"status": 200, "body": {"***": "ok"}},
        "served_models": [{"id": "m", "root": {"***": "r"}}],
    }
