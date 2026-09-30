# Receipt blocks shared by the image runs and the sweeps

Kind: reference. The serving-metrics and `server_args` receipt blocks. The
face, signature and check runs write them. The throughput sweeps write the
`server_args` block. Issue
[#347](https://github.com/Alberto-Codes/typevet/issues/347) moved these
sections to this page.

| Piece | Module |
|---|---|
| `/metrics` read, changes and gauges | `typevet_evals.serving_metrics` |
| `server_args` block | `typevet_evals.throughput.server_args` |

## Serving metrics block

The `server` key of the image-run `throughput` block holds the vLLM
`/metrics` changes over the run. It is `null` for llama.cpp. The
[LFW reference](eval-lfw-loader.md#throughput-block) lists the other
`throughput` keys.

The `server` block holds the `e2e`, `queue` and `prefill` histogram changes,
`prefix_cache_hit_rate`, `counters` and `gauges`. The `counters` are the
changes of `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries`,
`vllm:mm_cache_hits`, `vllm:mm_cache_queries`, `vllm:prompt_tokens`,
`vllm:prompt_tokens_cached`, `vllm:generation_tokens`,
`vllm:request_success` and `vllm:num_preemptions`. The `gauges` are
`vllm:kv_cache_usage_perc`, `vllm:num_requests_running` and
`vllm:num_requests_waiting`, read before and after the run. A gauge is a
point value, not a peak. A missing series is `unknown`. The two `/metrics`
reads are outside the wall time. Requests from other clients of the same
server also change the counters. The names come from vLLM `v0.30.0`.

## Server arguments block

Issue [#341](https://github.com/Alberto-Codes/typevet/issues/341) adds a
`server_args` key to the receipts that record serving metrics. These are the
face, signature and check receipts and the throughput sweep receipts. The key
tells a reader which vLLM server configuration made the receipt.
`typevet_evals.throughput.server_args` builds it.

| Key | Meaning |
|---|---|
| `cache_config` | Labels of the vLLM gauge `vllm:cache_config_info`, as strings; `null` when no reading holds the gauge |
| `cache_config_source` | Always `metrics`: the server reported `cache_config` on `/metrics` |
| `caller_stated` | Value of `TYPEVET_VLLM_SERVER_ARGS`, unchanged; `null` when the variable is not set |

The `cache_config` labels are the vLLM `CacheConfig` fields. Examples are
`block_size`, `enable_prefix_caching` and `gpu_memory_utilization`. The block
also keeps the `engine` label that the vLLM Prometheus logger adds. llama.cpp
has no such gauge, so its `cache_config` is `null`. A gauge sample can carry
a Prometheus timestamp after its value. The parser ignores the timestamp.

`/metrics` does not show the scheduler and model flags. Examples are
`--max-num-seqs`, `--max-num-batched-tokens` and `--logprobs-mode`. Only
`caller_stated` can hold them. It is what the caller stated, not what the
server reported. typevet does not parse or check it.

typevet does not read `/server_info`. That route needs
`VLLM_SERVER_DEV_MODE=1`, and the vLLM security guidance does not allow that
mode in production.

An image run takes `cache_config` from the `/metrics` reading before the run.
When that read fails, it uses the reading after the run. A throughput sweep
uses the first level reading that holds the gauge. Neither run adds a
`/metrics` read.
Receipts written before #341 have no `server_args` key.

## Related pages

- [LFW loader](eval-lfw-loader.md): the face-match run and its throughput
  block.
- [CEDAR loader](eval-cedar-loader.md): the signature-match run.
- [Synthetic checks](eval-synthetic-checks.md): the check-match run.
- [Performance on one H100](performance.md): the throughput sweep results.
