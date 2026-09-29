---
status: draft
---

# Typed-judgment release support matrix

Kind: reference.

Status: **draft**.

This page states what a **typed-judgment library release** supports today.
It lists the tested serving pins and the runtime limits.
It gives the commands that reproduce release evidence without a live model.
It also states what stays out of scope.
It implements the docs slice for [#191](https://github.com/Alberto-Codes/typevet/issues/191) under epic
[#108](https://github.com/Alberto-Codes/typevet/issues/108).
[#243](https://github.com/Alberto-Codes/typevet/issues/243) adds the tested vLLM pin.
This page does not replace the design contract on #191 or add new live measurements.

## Supported synchronous surface

| Area | Supported for release evidence | Notes |
|---|---|---|
| Typed tasks | `Noul`, `Choice`, `Score` via `JudgmentPort` | See [Native typed judgments](../explanation/native-typed-judgments.md) |
| Scoring path | `ScoringJudgmentAdapter` over `CandidateScoringPort` | Primary judgment transport on both backends |
| Generation path | `GenerationPort` with `LlamaCppGenerationAdapter` or `VllmGenerationAdapter` | llama.cpp uses `response_format`; vLLM uses `structured_outputs` |
| Offline proof | `typevet.testing` fakes (`ScriptedScoringFake`, …) | Default CI pyramid |
| Wheel consumer | Public imports only; no `tests.*` on install path | [#190](https://github.com/Alberto-Codes/typevet/issues/190) |
| Backend selection | `TYPEVET_BACKEND` selects `llama_cpp` (the default) or `vllm` | Other values raise `ValueError`. See [Configuration](configuration.md) |
| llama.cpp settings | `TYPEVET_LLAMA__*` via [`load_llama_settings`][typevet.adapters.inbound.load_llama_settings] | [Configuration](configuration.md) |
| vLLM settings | `TYPEVET_VLLM__*` via [`load_vllm_settings`][typevet.adapters.inbound.load_vllm_settings] | `BASE_URL` and `MODEL` are required. See [Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md) |

Library callers pass explicit adapter arguments. Importing `typevet` does not
read the environment.
The composition-root helpers `generation_adapter` and `open_judgment` read `TYPEVET_BACKEND`.
They build the llama.cpp or the vLLM adapter from it.

## Tested serving pins

Each row is one pinned configuration from one receipt.
A row is not a minimum version and says nothing about other quantizations or hardware.

| Backend | Server | Model and weights | Hardware | Receipt |
|---|---|---|---|---|
| vLLM | Stock `vllm/vllm-openai:v0.30.0` | `google/gemma-4-31B-it` revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16 | One H100 80 GB | [#170 receipt](https://github.com/Alberto-Codes/typevet/issues/170#issuecomment-5884707915) |
| llama.cpp (image input) | `llama-server` build `b11223-4da633776` | `gemma-4-31b-kv9-q4km-mm`, served template `native_gemma4_turn` | Not recorded | [#203 receipt](https://github.com/Alberto-Codes/typevet/issues/203#issuecomment-5882379255) |
| llama.cpp (generation) | `ghcr.io/ggml-org/llama.cpp:server-cuda-b11243` | `google/gemma-4-31B-it-qat-q4_0-gguf` revision `59dde24573e7e61570dba08b18a2e1fe246955ed`, Q4_0 | One A40 48 GB | [#129 receipt](https://github.com/Alberto-Codes/typevet/issues/129#issuecomment-5892208050) |

What each receipt shows:

- **vLLM (#170).** Every pre-registered gate passed in one run.
  Generation, PSAI image controls and CORD semantic acceptance passed.
  A malformed schema gave `BackendHttpError` after one POST.
  That run was at `abd496e`, before the schema check. At HEAD, that schema raises `ValueError` with no request.
  The server flags were `--max-model-len 8192`, `--max-num-seqs 4`, `--limit-mm-per-prompt {"image":2}` and `--logprobs-mode raw_logprobs`.
  The receipt covers this pin only.
  Setup steps are in [Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md).
- **llama.cpp image input (#203).** The CORD combined arm passed 5 of 5 semantic checks.
  The CLI exit code was 0.
  The sample is small (n 6 or 12 per check), so it is smoke evidence, not calibration.
- **llama.cpp generation (#129).** The grammar enforced string `enum`, integer bounds and the JSON object root.
  It did not enforce number bounds or `multipleOf`.
  typevet validates the returned value against the schema, and that check is the guard for number bounds.
  On that build, `multipleOf` gave empty output, which typevet raises as `GenerationError`.
  A malformed schema gave HTTP 500 on that build.
  This receipt ran at `2e23306`, before the adapters checked the schema and turned thinking off.

The vLLM and llama.cpp rows use different weights (BF16 and Q4 GGUF).
Do not attribute a result difference to the backend.

### Behaviour at HEAD on both backends

| Behaviour | Scope | Source |
|---|---|---|
| Schema check before any request | All four generation adapters. A schema that fails the Draft 2020-12 meta-schema raises `ValueError`. No request is sent. | `chat_completion.check_request_schema`, called in `llama_cpp.py`, `async_llama_cpp.py`, `vllm_generation.py` and `vllm_generation_async.py` |
| Thinking off in generation | llama.cpp and vLLM generation send `"chat_template_kwargs": {"enable_thinking": false}` | `llama_cpp.py`, `async_llama_cpp.py`, `vllm_generation.py`, `vllm_generation_async.py` (through `generation_body`) |
| Value check after generation | The returned JSON is validated against the request schema. A failure raises `SchemaValidationError`. | `chat_completion.validated_value` |
| Images in generation | vLLM generation sends images as `image_url` blocks. llama.cpp generation refuses a request with images before any HTTP call. | `vllm_content.py`, `LlamaCppGenerationAdapter._reject_media` |
| Native `Choice` capacity | The tokenizer sets the limit. Controls `"0"`, `"1"`, … must each be one token. | `bind_control_candidates` in `domain/judgment_normalize.py` ([#234](https://github.com/Alberto-Codes/typevet/issues/234)) |
| Async vLLM generation | One factory-built adapter per event loop. A call on a second loop raises `RuntimeError` before any request. | `async_vllm_generation_adapter`, `AsyncVllmGenerationAdapter` ([#224](https://github.com/Alberto-Codes/typevet/issues/224)) |

When a later control is not one token, the error states the capacity.
The message is `native Choice supports N options on this tokenizer; got M`.
On the Gemma 4 GGUF tokenizer that the #234 probe checked (llama.cpp), `"0"` to `"9"` are single tokens and `"10"` is two tokens.
Thus native `Choice` supports 10 options there. The vLLM tokenizer was not checked ([#234 probe](https://github.com/Alberto-Codes/typevet/issues/234#issuecomment-5897063711)).
`MAX_ENUM_CHOICES` stays at 24 for compiled schemas.

### Multimodal (primary: Gemma 4 native vision)

| Pin | Release-primary | Secondary (documented smokes) |
|---|---|---|
| Backend | Stock **llama.cpp** `llama-server`, or stock **vLLM** at the tested pin | Same |
| Model id (vision) | `gemma-4-31b-kv9-q4km-mm` on llama.cpp (example KV quant); served `gemma-4-31b-it` on vLLM | `gemma-3-4b-it-q4km-mm` in PSAI / legacy rows |
| Served template family | `native_gemma4_turn` on llama.cpp; vLLM applies its served chat template | `native_gemma3_turn` where a how-to still pins Gemma 3 |
| Env | `TYPEVET_LLAMA__MULTIMODAL_MODEL` on llama.cpp, or `TYPEVET_VLLM__MODEL` on vLLM | Same variables. On llama.cpp, the id must declare image input |
| Media type | `ImageInput` — PNG, JPEG, WebP; non-empty bytes | [Multimodal how-to](../how-to/run-a-multimodal-live-smoke.md) |

Text-only Gemma 4 on llama.cpp uses `TYPEVET_LLAMA__DEFAULT_MODEL` (or legacy
`TYPEVET_GEMMA_MODEL`). See [Run Gemma 4 on llama.cpp](../how-to/run-gemma4-llamacpp.md).

Task-specific smokes (not a single “release pass”):

- [CORD expense smoke](../how-to/run-the-cord-expense-smoke.md) — receipts + attachment floors ([#185](https://github.com/Alberto-Codes/typevet/issues/185), [#186](https://github.com/Alberto-Codes/typevet/issues/186)).
- [PSAI vision smoke](../how-to/run-the-psai-vision-smoke.md) — paired image controls ([#154](https://github.com/Alberto-Codes/typevet/issues/154)).
- [Small live judgment eval](../how-to/run-a-small-live-judgment-eval.md) — TPJEP eight-task path ([#133](https://github.com/Alberto-Codes/typevet/issues/133)).

## Runtime limits and ownership

| Limit | Where set | Release statement |
|---|---|---|
| HTTP deadline | `TYPEVET_LLAMA__TIMEOUT` or `TYPEVET_VLLM__TIMEOUT` (default 300 s each) | Callers and live tests may raise it (for example 900 s on multimodal smokes). The llama.cpp value reaches the client that the Gemma 4 native vision factory builds. A transport timeout during scoring raises `TransportError` ([`test_env_timeout_reaches_factory_http_client`, `test_transport_timeout_during_scoring_raises_transport_error`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_runtime_limits.py)) |
| Server URL | `TYPEVET_LLAMA__BASE_URL` or `TYPEVET_VLLM__BASE_URL` | llama.cpp default `http://127.0.0.1:8090`. vLLM has no default |
| Media marker | Cached per model id on `LlamaCppCandidateScoringAdapter` (llama.cpp only) | **Rebuild the adapter** after a router model reload. A reused adapter keeps the old marker; a new adapter reads the new marker. Stale markers fail tokenization — see multimodal how-to ([`test_reused_adapter_keeps_cached_marker_after_router_change`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_runtime_limits.py)). The vLLM factory calls no template probe |
| Image bytes / pixels | `ImageInput` validates mime and non-empty data only | **No** byte or pixel cap in domain types. An 8 MiB payload is accepted ([`test_image_input_accepts_eight_mib_payload`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_runtime_limits.py)). Pixel limits are not characterized ([#204](https://github.com/Alberto-Codes/typevet/issues/204)) |
| Images per request | `CandidateScoringRequest` requires one marker per image | **No** count cap in typevet. The scoring adapter sends every image of one request, for example 16 ([`test_scoring_sends_every_image_without_count_cap`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_runtime_limits.py)). The tested vLLM pin allowed 2 images per prompt |
| Native `Choice` options | Tokenizer, through `bind_control_candidates` | 10 options on the checked Gemma 4 GGUF tokenizer (vLLM not checked). More options raise `JudgmentValidationError` before any scoring call |
| Client closure | Factory ownership rules | The factory closes a client it owns and keeps a caller client open on every exit path ([`test_factory_http_client_ownership`](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_runtime_gemma_vision_factory.py), commit `71275a4`) |
| Concurrency / cancellation | `TYPEVET_VLLM__MAX_CONCURRENCY` (default 1) for the async vLLM adapter | Sets the POST limit for one adapter. Build one factory-built adapter per event loop. No async judgment surface ships (below) |
| Long-lived service | Not characterized beyond adapter lifetime rules | Do not infer production SLOs from smoke receipts ([#204](https://github.com/Alberto-Codes/typevet/issues/204)) |

## Failure classes (distinguish these)

| Class | Typical signal | Release evidence |
|---|---|---|
| Missing live config | pytest **skip** or CLI `skip:` stderr | Default; acceptable for dev |
| Missing live config (strict) | pytest **fail** with router/model reason | `TYPEVET_REQUIRE_LIVE=1` ([#191](https://github.com/Alberto-Codes/typevet/issues/191)) |
| Router / catalog | Skip or fail: unreachable, empty/invalid catalog, model not listed | [Live gate](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/evaluation/runner/live_gate.py) |
| Backend settings | `ValueError` for a bad `TYPEVET_BACKEND` or `TYPEVET_VLLM__*` value | Raised when the composition root reads the environment |
| Request / schema ask | `ValueError` (including `schema is not a valid JSON Schema:`), `TypeError`, `SchemaError` | Before any port call or request — [Errors](errors.md) |
| Native `Choice` capacity | `JudgmentValidationError`: `native Choice supports N options on this tokenizer; got M` | Before any scoring call ([#234](https://github.com/Alberto-Codes/typevet/issues/234)) |
| Event loop | `RuntimeError` on a second event loop for a factory-built async vLLM adapter | Before any request ([#224](https://github.com/Alberto-Codes/typevet/issues/224)) |
| Transport / backend | `TransportError`, `BackendHttpError`, `GenerationError` | Generation path |
| Scoring / attachment | `ScoringValidationError`, `ScoringUnsupportedCapabilityError`, attachment assert messages | [#185](https://github.com/Alberto-Codes/typevet/issues/185) floors |
| Template / capability mismatch | `JudgmentValidationError` (for example wrong `served_template`) | Multimodal and CORD harness |
| Semantic quality floors | `accept_combined_receipt` → `accepted=False` with named checks | [#184](https://github.com/Alberto-Codes/typevet/issues/184) / #161 — **not** the same as pytest live **pass** |
| Evidence identity drift | Immutable receipt write failure; digest mismatch in tests | [#186](https://github.com/Alberto-Codes/typevet/issues/186) |

### Vendored FAIL vs scratchpad PASS (honest)

- **Vendored CORD receipt** (`tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.json`):
  offline tests **pass** Gemma 4 capability and **attachment** gates ([#185](https://github.com/Alberto-Codes/typevet/issues/185)).
  The same receipt **fails** two #161 revision 1 **semantic** floors on the combined arm
  (`accept_combined_receipt` — see commands below). Identity metadata on that
  historical file is annotated in
  [`gemma4_kv9_direct_receipt.note.md`](https://github.com/Alberto-Codes/typevet/blob/main/tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.note.md).
- **Scratchpad live receipts** (`scratchpad/cord-expense/…`, `scratchpad/psai-vision/…`):
  a fresh live run can **pytest-pass** wiring and attachment.
  `accept_combined_receipt` can still **reject** the saved JSON for semantic floors.
  Smokes record metrics; they do **not** gate calibration or business accuracy.

Do not treat “live smoke passed once” as release acceptance without the
semantic and wheel commands in the next section.

## Executable evidence commands

Run from a checkout with `uv sync`. These commands do **not** require a live
model unless noted.

### Isolated wheel onboarding ([#190](https://github.com/Alberto-Codes/typevet/issues/190))

```bash
uv run pytest -q tests/contract/test_typed_judgment_wheel_onboarding.py
```

Matches the root [README](https://github.com/Alberto-Codes/typevet/blob/main/README.md) quick-start imports without
`PYTHONPATH=src`.

### Require-live gate (strict collection)

Default live tests **skip** when the router or model is missing. Release
evidence that must not skip:

```bash
export TYPEVET_REQUIRE_LIVE=1
uv run pytest tests/live/test_eval_runner_live.py -m live -q
```

Equivalent CLI flag: `--require-live` on `typevet.adapters.inbound.eval_cli` — see
[Live eval runner](eval-live-runner.md). Other live modules honor the same env
via [tests/live/gate.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/live/gate.py).

### Semantic acceptance on the designated CORD receipt

Offline unit proof on the pinned Gemma 4 direct receipt (attachment pass,
semantic fail on combined arm):

```bash
uv run pytest -q tests/unit/test_cord_expense_smoke_gemma4.py
```

The test `test_vendored_gemma4_combined_arm_fails_issue_161_answerable_and_contradicted`
calls `accept_combined_receipt` on
`tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.json`.

Broader semantic acceptance fixtures:

```bash
uv run pytest -q tests/unit/test_cord_semantic_acceptance.py
```

Operator acceptance on an exact saved receipt path (nonzero exit when
`accepted: false` or the file is malformed):

```bash
uv run python scripts/check_cord_semantic_acceptance.py \
  tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.json
echo $?   # expect 1 — historical FAIL on combined semantic floors

uv run python scripts/check_cord_semantic_acceptance.py \
  tests/fixtures/cord/semantic_acceptance/labeled_synthetic_pass.json
echo $?   # expect 0 — synthetic PASS exercising every floor

uv run python -m typevet.adapters.inbound.cord_semantic_acceptance_cli \
  tests/fixtures/cord/semantic_acceptance/gemma4_post187_combined_pass.json
echo $?   # expect 0 — vendored historical PASS (provenance in sibling .note.md)
```

Contract proof for the CLI exit codes:

```bash
uv run pytest -q tests/contract/test_cord_semantic_acceptance_cli.py
```

### Attachment and live wiring gates ([#185](https://github.com/Alberto-Codes/typevet/issues/185))

```bash
uv run pytest -q tests/unit/test_cord_expense_smoke_harness_gate.py \
  tests/unit/test_cord_expense_smoke_gemma4.py
```

Opt-in live orchestration (router + multimodal model required):

```bash
TYPEVET_LLAMA__MULTIMODAL_MODEL=gemma-4-31b-kv9-q4km-mm \
  TYPEVET_LLAMA__TIMEOUT=900 \
  uv run pytest tests/live/test_cord_expense_smoke_live.py -m live -q
```

Steps and failure table: [Run the CORD expense smoke](../how-to/run-the-cord-expense-smoke.md).

### vLLM acceptance harness ([#170](https://github.com/Alberto-Codes/typevet/issues/170))

Opt-in paid live run (vLLM server at the tested pin required):

```bash
TYPEVET_REQUIRE_LIVE=1 TYPEVET_BACKEND=vllm \
  TYPEVET_VLLM__BASE_URL=http://127.0.0.1:8000 \
  TYPEVET_VLLM__MODEL=gemma-4-31b-it \
  TYPEVET_VLLM__API_KEY="$VLLM_API_KEY" \
  TYPEVET_VLLM__USER_AGENT=curl/8.9.1 \
  TYPEVET_VLLM_RECEIPT=scratchpad/vllm/170-receipt.json \
  uv run pytest tests/live/test_vllm_acceptance_live.py -m live -q -s
```

The receipt path must not exist before the run.
[Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md) gives the server command.

### Experiment identity ([#186](https://github.com/Alberto-Codes/typevet/issues/186))

```bash
uv run pytest -q tests/unit/test_experiment_identity.py \
  tests/unit/test_experiment_identity_snapshot.py \
  tests/unit/test_cord_expense_call_accounting.py \
  tests/contract/test_cord_expense_receipt_snapshot.py
```

### Default non-live release floor

```bash
uv run pytest -q
uv run docvet check --all
```

Full gate table: [CLAUDE.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md).

## Deliberate exclusions

| Topic | Status | Tracker |
|---|---|---|
| Async judgment as a supported release API | Async generation adapters exist for llama.cpp and vLLM. There is no async `JudgmentPort` release claim | [Testing pyramid](testing.md) |
| vLLM beyond the tested pin | Other vLLM versions, models, precisions and GPUs are not tested | [#167](https://github.com/Alberto-Codes/typevet/issues/167) |
| vLLM throughput | One measured run only: one H100, one pin, short public texts, proxy in client latency. No general throughput claim. See [Performance on one H100](performance.md) | [#236](https://github.com/Alberto-Codes/typevet/issues/236) |
| Smoke pass ⇒ model quality / calibration | **Excluded** — smokes prove typed wiring, attachment, and controls | [#50](https://github.com/Alberto-Codes/typevet/issues/50), [#161](https://github.com/Alberto-Codes/typevet/issues/161) |
| TypeLLM / SGLang byte parity | Research references only | [Native typed judgments](../explanation/native-typed-judgments.md) |
| Shipped CLI / MCP composition root | Env-backed scripts and tests only | [Configuration](configuration.md) |

## Related pages

- [Supported imports](supported-imports.md)
- [Judgment live receipts](judgment-live-receipts.md) — measured rows with pins
- [Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md)
- [Verified evidence and inferred claims](../explanation/verification.md)
