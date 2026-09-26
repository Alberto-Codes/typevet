# Typed-judgment release support matrix

Kind: reference.

This page states what a **typed-judgment library release** supports today,
which commands reproduce release evidence without a live model, and what stays
out of scope. It implements the bounded docs slice for
[#191](https://github.com/Alberto-Codes/typevet/issues/191) under epic
[#108](https://github.com/Alberto-Codes/typevet/issues/108). It does not
replace the design contract on #191 or invent new live measurements.

## Supported synchronous surface

| Area | Supported for release evidence | Notes |
|---|---|---|
| Typed tasks | `Noul`, `Choice`, `Score` via `JudgmentPort` | See [Native typed judgments](../explanation/native-typed-judgments.md) |
| Scoring path | `ScoringJudgmentAdapter` over `CandidateScoringPort` | Primary judgment transport |
| Grammar-JSON path | `GenerationPort` + `LlamaCppGenerationAdapter` | [Live eval runner](eval-live-runner.md) slice only |
| Offline proof | `typevet.testing` fakes (`ScriptedScoringFake`, …) | Default CI pyramid |
| Wheel consumer | Public imports only; no `tests.*` on install path | [#190](https://github.com/Alberto-Codes/typevet/issues/190) |
| Composition root | `TYPEVET_LLAMA__*` via [`load_llama_settings`][] | [Configuration](configuration.md) |

Library callers pass explicit adapter arguments. Importing `typevet` does not
read the environment.

[`load_llama_settings`]: ../../src/typevet/adapters/inbound/settings.py

### Multimodal (primary: Gemma 4 native vision)

| Pin | Release-primary | Secondary (documented smokes) |
|---|---|---|
| Router backend | Stock **llama.cpp** `llama-server` | Same |
| Model id (vision) | `gemma-4-31b-kv9-q4km-mm` (example KV quant) | `gemma-3-4b-it-q4km-mm` in PSAI / legacy rows |
| Served template family | `native_gemma4_turn` | `native_gemma3_turn` where a how-to still pins Gemma 3 |
| Env | `TYPEVET_LLAMA__MULTIMODAL_MODEL` | Same variable; id must declare image input |
| Media type | `ImageInput` — PNG, JPEG, WebP; non-empty bytes | [Multimodal how-to](../how-to/run-a-multimodal-live-smoke.md) |

Text-only Gemma 4 uses `TYPEVET_LLAMA__DEFAULT_MODEL` (or legacy
`TYPEVET_GEMMA_MODEL`). See [Run Gemma 4 on llama.cpp](../how-to/run-gemma4-llamacpp.md).

Task-specific smokes (not a single “release pass”):

- [CORD expense smoke](../how-to/run-the-cord-expense-smoke.md) — receipts + attachment floors ([#185](https://github.com/Alberto-Codes/typevet/issues/185), [#186](https://github.com/Alberto-Codes/typevet/issues/186)).
- [PSAI vision smoke](../how-to/run-the-psai-vision-smoke.md) — paired image controls ([#154](https://github.com/Alberto-Codes/typevet/issues/154)).
- [Small live judgment eval](../how-to/run-a-small-live-judgment-eval.md) — TPJEP eight-task path ([#133](https://github.com/Alberto-Codes/typevet/issues/133)).

## Runtime limits and ownership

| Limit | Where set | Release statement |
|---|---|---|
| HTTP deadline | `TYPEVET_LLAMA__TIMEOUT` (default 300 s) | Callers and live tests may raise it (for example 900 s on multimodal smokes) |
| Router URL | `TYPEVET_LLAMA__BASE_URL` | Default `http://127.0.0.1:8090` |
| Media marker | Cached per model id on `LlamaCppCandidateScoringAdapter` | **Rebuild the adapter** after a router model reload; stale markers fail tokenization — see multimodal how-to |
| Image bytes / pixels | `ImageInput` validates mime and non-empty data only | **No** documented byte, pixel, or per-request image-count cap in domain types ([#191](https://github.com/Alberto-Codes/typevet/issues/191) open gap) |
| Concurrency / cancellation | Caller-owned httpx client lifecycle | No shipped async judgment release surface (below) |
| Long-lived service | Not characterized beyond adapter lifetime rules | Do not infer production SLOs from smoke receipts |

## Failure classes (distinguish these)

| Class | Typical signal | Release evidence |
|---|---|---|
| Missing live config | pytest **skip** or CLI `skip:` stderr | Default; acceptable for dev |
| Missing live config (strict) | pytest **fail** with router/model reason | `TYPEVET_REQUIRE_LIVE=1` ([#191](https://github.com/Alberto-Codes/typevet/issues/191)) |
| Router / catalog | Skip or fail: unreachable, empty/invalid catalog, model not listed | [Live gate](../../src/typevet/evaluation/runner/live_gate.py) |
| Request / schema ask | `ValueError`, `TypeError`, `SchemaError` | Before any port call — [Errors](errors.md) |
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
  [`gemma4_kv9_direct_receipt.note.md`](../../tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.note.md).
- **Scratchpad live receipts** (`scratchpad/cord-expense/…`, `scratchpad/psai-vision/…`):
  a fresh live run can **pytest-pass** wiring and attachment while
  `accept_combined_receipt` still **rejects** the saved JSON for semantic floors.
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

Matches the root [README](../../README.md) quick-start imports without
`PYTHONPATH=src`.

### Require-live gate (strict collection)

Default live tests **skip** when the router or model is missing. Release
evidence that must not skip:

```bash
export TYPEVET_REQUIRE_LIVE=1
uv run pytest tests/live/test_eval_runner_live.py -m live -q
```

Equivalent CLI flag: `--require-live` on `typevet.eval_runner_cli` — see
[Live eval runner](eval-live-runner.md). Other live modules honor the same env
via [tests/live/gate.py](../../tests/live/gate.py).

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

uv run python -m typevet.cord_semantic_acceptance_cli \
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

Full gate table: [CLAUDE.md](../../CLAUDE.md).

## Deliberate exclusions

| Topic | Status | Tracker |
|---|---|---|
| Async judgment as a supported release API | `AsyncGenerationPort` exists for generation parity tests only; no async `JudgmentPort` release claim | [Testing pyramid](testing.md) |
| vLLM (or non–llama.cpp) serving | Not in this release matrix | [#167](https://github.com/Alberto-Codes/typevet/issues/167) |
| Smoke pass ⇒ model quality / calibration | **Excluded** — smokes prove typed wiring, attachment, and controls | [#50](https://github.com/Alberto-Codes/typevet/issues/50), [#161](https://github.com/Alberto-Codes/typevet/issues/161) |
| TypeLLM / SGLang byte parity | Research references only | [Native typed judgments](../explanation/native-typed-judgments.md) |
| Shipped CLI / MCP composition root | Env-backed scripts and tests only | [Configuration](configuration.md) |

## Related pages

- [Supported imports](supported-imports.md)
- [Judgment live receipts](judgment-live-receipts.md) — measured rows with pins
- [Verified evidence and inferred claims](../explanation/verification.md)
