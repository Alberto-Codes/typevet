"""Unit tests for the vLLM acceptance transport module split (#229)."""

from __future__ import annotations

import httpx
import pytest

from typevet_evals.vllm_acceptance import core as vllm_acceptance
from typevet_evals.vllm_acceptance.transport import (
    CountingTransport,
    kv_cache_usage,
)

pytestmark = pytest.mark.unit


def test_vllm_acceptance_re_exports_the_moved_names() -> None:
    assert vllm_acceptance.CountingTransport is CountingTransport
    assert vllm_acceptance.kv_cache_usage is kv_cache_usage


def test_kv_cache_usage_reads_the_gauge_from_the_new_module() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="vllm:kv_cache_usage_perc 0.5\n")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://vllm") as client:
        assert kv_cache_usage(client) == 0.5
