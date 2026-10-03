# JevBench runner

Kind: reference. This command evaluates public JevBench tasks through the existing judgevet provider bridge.

The [Gemma 4 public baseline](../explanation/jevbench-public-baseline.md) records one complete live run and its limits.

## Source identity

The eval workspace pins [JevBench commit bb05a335](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc).
The full revision is `bb05a335bc809e61b20c0f745d25499a82b326fc`.
Upstream package version strings disagree. Identify this harness by its commit.
Historical v1.4.2.2 board values do not describe this pin.

The public original dataset contains 72 tasks at this revision.
The download command fixes the dataset revision:

```bash
mkdir -p scratchpad/jevbench-data
curl --fail --location \
  https://raw.githubusercontent.com/fstandhartinger/jevbench/bb05a335bc809e61b20c0f745d25499a82b326fc/datasets/public/original.jsonl \
  --output scratchpad/jevbench-data/original.jsonl
sha256sum scratchpad/jevbench-data/original.jsonl
uv sync
```

The expected SHA-256 is `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180`.

## Commands

Each run requires fresh results, raw and manifest paths.
Paths must remain outside the installed upstream package.
The ledger can persist across runs to enforce a shared reservation cap.

Offline smoke command:

```bash
TYPEVET_BACKEND=fake uv run python -m typevet_evals.jevbench_run \
  --tasks scratchpad/jevbench-data/original.jsonl --model fake --limit 3 \
  --results scratchpad/jevbench-fake/results.jsonl \
  --raw-dir scratchpad/jevbench-fake/raw \
  --ledger scratchpad/jevbench-fake/ledger.jsonl \
  --manifest scratchpad/jevbench-fake/manifest.json
```

Gemma command for an existing llama.cpp server:

```bash
TYPEVET_BACKEND=llama_cpp \
TYPEVET_LLAMA__BASE_URL=http://127.0.0.1:8080 \
TYPEVET_LLAMA__MULTIMODAL_MODEL=gemma-4-31b-kv9-q4km-mm \
uv run python -m typevet_evals.jevbench_run \
  --tasks scratchpad/jevbench-data/original.jsonl \
  --model gemma-4-31b-kv9-q4km-mm \
  --results scratchpad/jevbench-gemma/results.jsonl \
  --raw-dir scratchpad/jevbench-gemma/raw \
  --ledger scratchpad/jevbench-gemma/ledger.jsonl \
  --manifest scratchpad/jevbench-gemma/manifest.json
```

Replace the server URL and both model identifiers with the loaded service configuration.
The native factory reads `TYPEVET_LLAMA__MULTIMODAL_MODEL`.
`TYPEVET_LLAMA__DEFAULT_MODEL` alone does not configure that factory.
The alias does not identify model weights or their quantization.
Record those separately for live evidence.
The offline tests do not establish live model quality.

Upstream summary command:

```bash
uv run python -m jevbench.cli summarize \
  --tasks scratchpad/jevbench-data/original.jsonl \
  --results scratchpad/jevbench-gemma/results.jsonl \
  --ledger scratchpad/jevbench-gemma/ledger.jsonl \
  > scratchpad/jevbench-gemma/summary.json
```

## Options and evidence

| Option | Meaning | Default |
|---|---|---|
| `--tasks` | Canonical upstream JSONL dataset | Required |
| `--model` | Requested provider model | Required |
| `--results` | Fresh upstream result JSONL file | Required |
| `--raw-dir` | Fresh directory for upstream raw JSON evidence | Required |
| `--ledger` | Shared upstream reservation ledger | Required |
| `--manifest` | Fresh run identity JSON file | Required |
| `--limit` | Positive task count cap in dataset order | All tasks |
| `--cap-usd` | Upstream reservation cap in USD | 15 |
| `--reserve-usd` | Reservation per attempt in USD | 0.02 |

The manifest records the upstream commit, dataset SHA-256, model identifiers, counts and non-secret run options.
The requested count reflects `--limit`. The dataset count covers the input file.
The resolved model identifiers come from provider responses. The session model records the configured typevet session.
A failed or incomplete run returns a nonzero process status.

Upstream owns scoring, raw persistence, timing, reservation settlement and stop rules.
The adapter makes no retries and invents no fallback distributions.
Unknown tariffs keep `cost_usd` null and `cost_basis` unknown.
`charged_usd` can retain a reservation. It is not measured spend.

## Mapping

Requests contain only `state`, configured `model` and `questions.decision`.
The decision includes `type`, verbatim `instructions` and non-null `criteria`.
Gold, provenance and other task metadata never enter the provider request.

| Kind | Provider answer | Upstream probabilities |
|---|---|---|
| Noul | Typed probability `p` | `no: 1-p`, `yes: p` |
| Choice | Typed distribution | Exact task labels, including case and spaces |
| Score | Integer-indexed typed distribution | Decimal strings of zero-based indices |

Score criteria are ordered levels. The answer legend must match them.
The scalar score never supplies a class label. Upstream calculates the class and ordinal expected value.
Rounded distributions remain unchanged. Upstream applies its strict 0.001 and headline 0.02 tolerances.
`probs_source=native` identifies this route. It makes no calibration claim.
Known failures retain safe codes or exception classes, without exception messages.
Only Jev errors can carry an available HTTP status. In-process success has no HTTP status.
