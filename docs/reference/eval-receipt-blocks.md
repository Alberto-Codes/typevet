# Receipt blocks shared by the eval runs

Kind: reference. The serving-metrics, `server_args` and `wording_digests`
receipt blocks. The face, signature and check runs write the first two. The
throughput sweeps write the `server_args` block. The wording receipts write
the `wording_digests` block. Issue
[#347](https://github.com/Alberto-Codes/typevet/issues/347) moved the first
two sections to this page.

| Piece | Module |
|---|---|
| `/metrics` read, changes and gauges | `typevet_evals.serving_metrics` |
| `server_args` block | `typevet_evals.throughput.server_args` |
| `wording_digests` block | `typevet_evals.wording.digests` |

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

## Wording digests block

Issue [#362](https://github.com/Alberto-Codes/typevet/issues/362) adds a
`wording_digests` key to the held-out receipt (#309) and the comparison
receipt (#329). A reader matches a receipt to a candidate by digest. A
reader does not need to compare the full text. The receipts keep
`seed_text` and `evolved_text` verbatim.

A candidate is a mapping of component name to text. Each receipt digests two
mappings. `seed` is `{"instructions": seed_text}`. `evolved` is
`{"instructions": evolved_text}`. Each arm holds these keys:

| Key | Meaning |
|---|---|
| `components` | Component name to the SHA-256 hex digest of its UTF-8 text |
| `mapping` | SHA-256 hex digest of the canonical JSON of the whole mapping |
| `gepa_candidate_id` | gepa-adk `Candidate.id` of a candidate with the same components |

The shape:

```json
{
  "seed": {
    "components": {"instructions": "<64 hex>"},
    "mapping": "<64 hex>",
    "gepa_candidate_id": "<12 hex>"
  },
  "evolved": {
    "components": {"instructions": "<64 hex>"},
    "mapping": "<64 hex>",
    "gepa_candidate_id": "<12 hex>"
  }
}
```

The digest rules, in Python:

| Digest | Rule |
|---|---|
| component | `sha256(text.encode("utf-8")).hexdigest()` |
| `mapping` | `sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()` |
| `gepa_candidate_id` | `sha256(json.dumps(m, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]` |

The `mapping` rule uses the Python default and escapes non-ASCII text as
`\uXXXX`. A tool that recomputes it outside Python must do the same.
The `mapping` digest is not equal to gepa-adk `Candidate.id`. The gepa-adk
rule keeps the default JSON spaces and does not escape non-ASCII text. It
also keeps only 12 hex characters. The block therefore records both values.
gepa-adk `Candidate.id` is in `gepa_adk/domain/models.py`.

The `gepa_candidate_id` value depends on the component names. The wording
run names its gepa-adk component by the question key, for example
`is_scam`. The receipt names it `instructions`. Thus `gepa_candidate_id`
does not equal the `candidate_id` that the gepa-adk engine logs for that
run.

The seed `criteria` are not in the mapping. They are a label-to-text
mapping, not one text, and the evolution does not change them. A later
multi-part candidate adds one `components` entry for each part.
Receipts written before #362 have no `wording_digests` key.

## Related pages

- [LFW loader](eval-lfw-loader.md): the face-match run and its throughput
  block.
- [CEDAR loader](eval-cedar-loader.md): the signature-match run.
- [Synthetic checks](eval-synthetic-checks.md): the check-match run.
- [Performance on one H100](performance.md): the throughput sweep results.
