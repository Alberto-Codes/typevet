"""Offline tests for the ``server_args`` receipt block (#341).

The ``/metrics`` text is the shared fixture
``evals/tests/fixtures/vllm_cache_config_metrics.txt``. No test reads the
network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet_evals.throughput.server_args import (
    CACHE_CONFIG_SOURCE,
    SERVER_ARGS_ENV,
    cache_config,
    server_args_block,
    stated_server_args,
)

pytestmark = pytest.mark.unit

CACHE_CONFIG_TEXT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "vllm_cache_config_metrics.txt"
).read_text(encoding="utf-8")
CACHE_CONFIG: dict[str, str] = {
    "block_size": "16",
    "cache_dtype": "auto",
    "calculate_kv_scales": "False",
    "cpu_offload_gb": "0",
    "enable_prefix_caching": "True",
    "engine": "0",
    "gpu_memory_utilization": "0.9",
    "num_gpu_blocks": "4096",
    "sliding_window": "None",
}
STATED = "--max-num-seqs 64 --logprobs-mode processed_logprobs"


def test_labels_parse_from_the_shared_fixture_as_strings() -> None:
    assert cache_config(CACHE_CONFIG_TEXT) == CACHE_CONFIG


def test_missing_gauge_or_reading_is_none() -> None:
    text = 'vllm:num_requests_running{engine="0",model_name="m"} 0.0\n'
    assert cache_config(text) is None
    assert cache_config(None) is None
    assert cache_config("") is None


def test_first_sample_wins_and_escapes_are_read() -> None:
    text = (
        'vllm:cache_config_info_extra{block_size="99"} 1.0\n'
        'vllm:cache_config_info{a="x,y}z",b="say \\"hi\\"",c="back\\\\slash"} 1.0\n'
        'vllm:cache_config_info{a="second"} 1.0\n'
    )
    assert cache_config(text) == {"a": "x,y}z", "b": 'say "hi"', "c": "back\\slash"}


def test_block_marks_observed_and_stated_values() -> None:
    assert server_args_block(CACHE_CONFIG, STATED) == {
        "cache_config": CACHE_CONFIG,
        "cache_config_source": CACHE_CONFIG_SOURCE,
        "caller_stated": STATED,
    }
    assert server_args_block(None, None) == {
        "cache_config": None,
        "cache_config_source": "metrics",
        "caller_stated": None,
    }


def test_stated_args_are_read_verbatim() -> None:
    raw = f"  {STATED}  "
    assert SERVER_ARGS_ENV == "TYPEVET_VLLM_SERVER_ARGS"
    assert stated_server_args({SERVER_ARGS_ENV: raw}) == raw
    assert stated_server_args({}) is None
