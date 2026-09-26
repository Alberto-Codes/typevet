"""Shared schema and helpers for sync/async outbound contract parity."""

from __future__ import annotations

CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}
