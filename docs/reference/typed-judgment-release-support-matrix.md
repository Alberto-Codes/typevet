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
| vLLM | Stock `vllm/vllm-openai:v0.30.0` | `google/gemma-4-31B-it` revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16 | One H100 80 GB | [#170 receipt](https://github.com/Alberto-Codes/typevet/issues/170#issuecomment-5884707915); KV-cache gauge: [#231 receipt](https://github.com/Alberto-Codes/typevet/issues/231#issuecomment-5904832873) |
| llama.cpp (image input) | `llama-server` build `b11223-4da633776` | Alias `gemma-4-31b-kv9-q4km-mm`, served template `native_gemma4_turn`. The alias loads `gemma-4-31b-24gib-kv9-decoder.gguf`, ftype `Q2_K - Medium` (not Q4_K_M), 16.0 GB, with a `--chat-template-file` override on image `server-cuda-b11243` ([#233](https://github.com/Alberto-Codes/typevet/issues/233)) | Not recorded | [#203 receipt](https://github.com/Alberto-Codes/typevet/issues/203#issuecomment-5882379255) |
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

The rows use different weights: BF16 on vLLM, a Q4_0 GGUF for llama.cpp generation and a `Q2_K - Medium` GGUF for llama.cpp image input.
Do not attribute a result difference to the backend.

### Behaviour at HEAD on both backends

| Behaviour | Scope | Source |
|---|---|---|
| Schema check before any request | All four generation adapters. A schema that fails the Draft 2020-12 meta-schema raises `ValueError`. No request is sent. | `chat_completion.check_request_schema`, called in `llama_cpp/generation.py`, `llama_cpp/generation_async.py`, `vllm/generation.py` and `vllm/generation_async.py` |
| Thinking off in generation | llama.cpp and vLLM generation send `"chat_template_kwargs": {"enable_thinking": false}` | `llama_cpp/generation.py`, `llama_cpp/generation_async.py`, `vllm/generation.py`, `vllm/generation_async.py` (through `generation_body`) |
| Value check after generation | The returned JSON is validated against the request schema. A failure raises `SchemaValidationError`. | `chat_completion.validated_value` |
| Images in generation | vLLM generation sends images as `image_url` blocks. llama.cpp generation refuses a request with images before any HTTP call. | `vllm/content.py`, `LlamaCppGenerationAdapter._reject_media` |
| Native `Choice` capacity | Execute accepts 24 options at most (`MAX_ENUM_CHOICES`). Controls are `"0"` to `"9"`, then `"A"` to `"Z"`, so binding can label 36. The tokenizer can set a lower limit: each control must be one token. | `bind_control_candidates` in `domain/judgment_normalize.py` ([#234](https://github.com/Alberto-Codes/typevet/issues/234)) |
| Off-option mass in scoring | `CandidateScoringResult.off_option_mass` is the probability mass outside the candidate tokens: 1 minus the sum of the raw candidate probabilities. llama.cpp reports it when the response holds exactly `n_vocab` entries and their total is within 0.01 of 1, else `None`. `n_vocab` is `meta.n_vocab` of the model entry in `/v1/models`. The adapter keeps a known size. It stops after 3 reads per model that give no size ([#321](https://github.com/Alberto-Codes/typevet/issues/321)). Router build `b11277-eae11d221` reports 262144 for the loaded Gemma 4 31B Q2_K file and no `meta` for an unloaded model. The adapter reads it after the first `/completion` call, which loads the model, and sends it as `n_probs` on later calls. An unknown size gives `None`. A caller can set `n_vocab`; the adapter then trusts that value and sends no `/v1/models` request. The Gemma native vision factory sets 262144 by default. vLLM always reports `None`, because the response holds at most 128 token ids. One local check on the Q2_K Gemma 4 file (review, 2026-09-30) returned 262144 entries that summed to 1.0006. No live test exercises the field. An opt-in `off_option_threshold` on `judge` in `[0, 1]` sets `off_option_flag` on each answer whose mass is above it. The default `None` turns the guard off. The guard flags and does not raise. `JudgmentResponse.off_option` keeps an `OffOptionReceipt` per answer: `off_option_mass`, `off_option_threshold` and `off_option_flag`. A `None` mass never flags; the receipt records `off_option_mass: null`. `JudgmentPort.judge` declares the keyword, so a caller typed to the port can pass it ([#368](https://github.com/Alberto-Codes/typevet/issues/368)). The scoring adapter behind each factory session applies it. `CalibratedJudgment`, the vLLM key-masking and request-id wrappers and the evals ledger wrappers forward it. The judgevet bridge accepts it on `system_one`, the async `system_one` and `system_one_media` and forwards it. No judgevet setting maps to it, and the judgevet response has no field for the receipt. | `_off_option_mass` in `llama_cpp/scoring.py`, `llama_cpp/vocabulary.py`, `vllm/scoring.py`; `apply_off_option_threshold` in `domain/decision_execute.py`; `ports/judgment.py` ([`test_off_option_mass.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_off_option_mass.py), [`test_llama_cpp_vocabulary.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_llama_cpp_vocabulary.py), [`test_off_option_guard.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_off_option_guard.py), [`test_off_option_guard_agreement.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_off_option_guard_agreement.py), [`test_off_option_threshold_forwarding.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_off_option_threshold_forwarding.py), [#297](https://github.com/Alberto-Codes/typevet/issues/297), [#353](https://github.com/Alberto-Codes/typevet/issues/353)) |
| Request id in the judgment receipt | vLLM only. With `TYPEVET_VLLM__REQUEST_ID_HEADER` set, `JudgmentResponse.request_ids` maps each question name to the id in its scoring request. When a scoring wrapper sends more than one request for a question, the last id is kept. The `/tokenize` ids are not in it. Without the header, `request_ids` is empty. A vLLM `BackendHttpError` or `TransportError` holds the id of the failed request as `request_id`, else `None`. typevet never logs the id and does not mask it. A llama.cpp response keeps an empty `request_ids`. Contract tests on `httpx.MockTransport` prove it; no live call does. | `vllm/request_ids.py`, `vllm/http_mapping.py`, `gateway_headers.py` ([`test_vllm_request_ids.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_vllm_request_ids.py), [#356](https://github.com/Alberto-Codes/typevet/issues/356)) |
| Async vLLM generation | One factory-built adapter per event loop. A call on a second loop raises `RuntimeError` before any request. | `async_vllm_generation_adapter`, `AsyncVllmGenerationAdapter` ([#224](https://github.com/Alberto-Codes/typevet/issues/224)) |

When a later control is not one token, the error states the capacity.
The message is `native Choice supports N options on this tokenizer; got M`.
25 to 36 options raise `DecisionExecutionError`: `choice count must be between 2 and 24, got M`.
More than 36 options raise `native Choice supports at most 36 options; got M`.
The #286 check tokenized `"0"` to `"9"` and `"A"` to `"Z"` on the local Gemma 4 GGUF tokenizer (llama.cpp) and on the cached Hugging Face tokenizer for vLLM. Each is a single token, with the same ids on both ([#286](https://github.com/Alberto-Codes/typevet/issues/286)).
Thus control binding can label 36 options there, and the usable limit is the execute limit of 24. Other tokenizers can support fewer.
The #299 check did the same on the `allenai/Molmo2-4B` tokenizer (revision `042abfa7`, tokenizer files only). Each control is one token after the ChatML assistant header, so the limit there is also 24. No Molmo2 model call was made ([#299](https://github.com/Alberto-Codes/typevet/issues/299)).
The Hugging Face check used the QAT checkpoint tokenizer, not the exact BF16 serving pin.
No calibration receipt exists for more than 10 options.
One live 24-option run on the local Q2_K Gemma 4 pin gave a valid distribution ([#288](https://github.com/Alberto-Codes/typevet/issues/288)). One run is not calibration.
The eval code computes top-label ECE and class-wise ECE for any label set. The public workloads accept 24 options ([#296](https://github.com/Alberto-Codes/typevet/issues/296)).
A class with fewer than 30 gold instances gets the status "insufficient N" and is not in the class-wise mean. No live run has used these metrics yet.
`MAX_ENUM_CHOICES` stays at 24 for compiled schemas.

### Multimodal (primary: Gemma 4 native vision)

| Pin | Release-primary | Secondary (documented smokes) |
|---|---|---|
| Backend | Stock **llama.cpp** `llama-server`, or stock **vLLM** at the tested pin | Same |
| Model id (vision) | `gemma-4-31b-kv9-q4km-mm` on llama.cpp. The alias loads `gemma-4-31b-24gib-kv9-decoder.gguf`, ftype `Q2_K - Medium` (not Q4_K_M), 16.0 GB, image `server-cuda-b11243`, with a `--chat-template-file` override ([#233](https://github.com/Alberto-Codes/typevet/issues/233)). Served `gemma-4-31b-it` on vLLM | `gemma-3-4b-it-q4km-mm` in PSAI / legacy rows |
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
| Media marker | Cached per model id on `LlamaCppCandidateScoringAdapter` (llama.cpp only) | A router model reload changes the marker. A successful request keeps the cached marker and sends no extra call ([`test_reused_adapter_keeps_cached_marker_after_router_change`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_runtime_limits.py)). When a media request gets HTTP 400 `Failed to tokenize prompt`, the adapter reads `/props` once, rebuilds the prompt and sends it once more. A second failure raises `BackendHttpError` ([`test_stale_marker_refreshes_once_and_retries_once`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_llama_cpp_media_marker_refresh.py), [#322](https://github.com/Alberto-Codes/typevet/issues/322)). The vLLM factory calls no template probe |
| Image bytes / pixels | `ImageInput` validates mime and non-empty data only | **No** byte or pixel cap in domain types. An 8 MiB payload is accepted ([`test_image_input_accepts_eight_mib_payload`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_runtime_limits.py)). One live run on the llama.cpp vision alias sent synthetic PNGs of 256, 1024, 2048 and 4096 px squares, plus 64x4096 and 4096x64. Each rung returned a valid `Noul` answer within the time budget. Prompt tokens stopped at 1160 from 2048 px up. The receipt does not show the cause. One run is not a limit guarantee ([`test_image_pixel_limits_live.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/live/test_image_pixel_limits_live.py), [receipt](https://github.com/Alberto-Codes/typevet/blob/main/tests/fixtures/runtime_limits/pixel_limits_llama_cpp_receipt.json), [#204](https://github.com/Alberto-Codes/typevet/issues/204)) |
| Images per request | `CandidateScoringRequest` requires one marker per image | **No** count cap in typevet. The scoring adapter sends every image of one request, for example 16 ([`test_scoring_sends_every_image_without_count_cap`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_runtime_limits.py)). The tested vLLM pin allowed 2 images per prompt |
| Native `Choice` options | Tokenizer, through `bind_control_candidates` | 24 options, the execute limit `MAX_ENUM_CHOICES`. Control binding can label 36 options on the checked Gemma 4 tokenizers (local GGUF and cached Hugging Face, [#286](https://github.com/Alberto-Codes/typevet/issues/286)). The binding limit depends on the tokenizer. 25 to 36 options raise `DecisionExecutionError`, and more than 36 raise `JudgmentValidationError`, before that question's scoring call. No calibration receipt for more than 10 options. One live 24-option run on the local pin gave a valid distribution ([#288](https://github.com/Alberto-Codes/typevet/issues/288)). The eval code computes top-label and class-wise ECE for any label set. The public workloads accept 24 options. No live run has used these metrics yet ([#296](https://github.com/Alberto-Codes/typevet/issues/296)) |
| Client closure | Factory ownership rules | The factory closes a client it owns and keeps a caller client open on every exit path ([`test_factory_http_client_ownership`](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_runtime_gemma_vision_factory.py), commit `71275a4`) |
| Concurrency / cancellation | `TYPEVET_VLLM__MAX_CONCURRENCY` (default 1) for the async vLLM adapter | Sets the POST limit for one adapter. Build one factory-built adapter per event loop. No async judgment surface ships (below) |
| Long-lived service | One Gemma 4 native vision session on a llama.cpp router | Offline stub tests: 200 sequential calls on one session stay valid and use one pooled connection. After a router restart, the session opens a new connection. The first call may raise `TransportError`, and the calls after it succeed. A call while the router is down raises `TransportError`. An error status on the factory `/tokenize` call raises `BackendHttpError` ([#298](https://github.com/Alberto-Codes/typevet/issues/298)). A 200 `/tokenize` body without a `tokens` list raises `GenerationError` ([#310](https://github.com/Alberto-Codes/typevet/issues/310)). The router can close a connection (new or reused) before a response head. Then `/tokenize`, `/props` and `/completion` send the request once more on a new connection. A second close raises `TransportError` ([`test_llama_cpp_keepalive_retry.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_llama_cpp_keepalive_retry.py), [#305](https://github.com/Alberto-Codes/typevet/issues/305)). Do not infer production SLOs from these tests ([`test_long_lived_adapter.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_long_lived_adapter.py), [#204](https://github.com/Alberto-Codes/typevet/issues/204)) |

## Failure classes (distinguish these)

| Class | Typical signal | Release evidence |
|---|---|---|
| Missing live config | pytest **skip** or CLI `skip:` stderr | Default; acceptable for dev |
| Missing live config (strict) | pytest **fail** with router/model reason | `TYPEVET_REQUIRE_LIVE=1` ([#191](https://github.com/Alberto-Codes/typevet/issues/191)) |
| Router / catalog | Skip or fail: unreachable, empty/invalid catalog, model not listed | [Live gate](https://github.com/Alberto-Codes/typevet/blob/main/evals/src/typevet_evals/runner/live_gate.py) |
| Backend settings | `ValueError` for a bad `TYPEVET_BACKEND` or `TYPEVET_VLLM__*` value | Raised when the composition root reads the environment |
| Request / schema ask | `ValueError` (including `schema is not a valid JSON Schema:`), `TypeError`, `SchemaError` | Before any port call or request — [Errors](errors.md) |
| Native `Choice` capacity | `JudgmentValidationError`: `native Choice supports N options on this tokenizer; got M`, or `native Choice supports at most 36 options; got M` for more than 36 options | Before any scoring call ([#234](https://github.com/Alberto-Codes/typevet/issues/234)) |
| Native `Choice` execute limit | `DecisionExecutionError`: `choice count must be between 2 and 24, got M` for 25 to 36 options | Before any scoring call ([#287](https://github.com/Alberto-Codes/typevet/issues/287)) |
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
uv run pytest evals/tests/live/test_eval_runner_live.py -m live -q
```

Equivalent CLI flag: `--require-live` on `typevet_evals.cli.eval_runner` — see
[Live eval runner](eval-live-runner.md). Other live modules honor the same env
via [tests/live/gate.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/live/gate.py).

### Semantic acceptance on the designated CORD receipt

Offline unit proof on the pinned Gemma 4 direct receipt (attachment pass,
semantic fail on combined arm):

```bash
uv run pytest -q evals/tests/unit/test_cord_expense_smoke_gemma4.py
```

The test `test_vendored_gemma4_combined_arm_fails_issue_161_answerable_and_contradicted`
calls `accept_combined_receipt` on
`tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.json`.

Broader semantic acceptance fixtures:

```bash
uv run pytest -q evals/tests/unit/test_cord_semantic_acceptance.py
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

uv run python -m typevet_evals.cli.cord_semantic_acceptance \
  tests/fixtures/cord/semantic_acceptance/gemma4_post187_combined_pass.json
echo $?   # expect 0 — vendored historical PASS (provenance in sibling .note.md)
```

Contract proof for the CLI exit codes:

```bash
uv run pytest -q evals/tests/contract/test_cord_semantic_acceptance_cli.py
```

### Attachment and live wiring gates ([#185](https://github.com/Alberto-Codes/typevet/issues/185))

```bash
uv run pytest -q evals/tests/unit/test_cord_expense_smoke_harness_gate.py \
  evals/tests/unit/test_cord_expense_smoke_gemma4.py
```

Opt-in live orchestration (router + multimodal model required):

```bash
TYPEVET_LLAMA__MULTIMODAL_MODEL=gemma-4-31b-kv9-q4km-mm \
  TYPEVET_LLAMA__TIMEOUT=900 \
  uv run pytest evals/tests/live/test_cord_expense_smoke_live.py -m live -q
```

The alias `gemma-4-31b-kv9-q4km-mm` loads `gemma-4-31b-24gib-kv9-decoder.gguf`.
That file is 16.0 GB with ftype `Q2_K - Medium`, not Q4_K_M.
The router uses a `--chat-template-file` override on image `server-cuda-b11243` ([#233](https://github.com/Alberto-Codes/typevet/issues/233)).

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
  uv run pytest evals/tests/live/test_vllm_acceptance_live.py -m live -q -s
```

The receipt path must not exist before the run.
[Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md) gives the server command.

### Experiment identity ([#186](https://github.com/Alberto-Codes/typevet/issues/186))

```bash
uv run pytest -q evals/tests/unit/test_experiment_identity.py \
  evals/tests/unit/test_experiment_identity_snapshot.py \
  evals/tests/unit/test_cord_expense_call_accounting.py \
  evals/tests/contract/test_cord_expense_receipt_snapshot.py
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
