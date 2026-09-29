# TypeLLM, Jev and what typevet must preserve

Kind: explanation.

TypeLLM is inspired by TypeSafe’s System One / Jev. judgevet is a typed
client for that same Jev surface. typevet’s long game is local type-safe
generation that can serve callers like judgevet without SGLang. Grammar-JSON
generation is the transport floor. Categorical decisions scored from candidate
logprobs are the part of the TypeLLM decision model that typevet ships today.

## Three products, one family

| Piece | Role |
|---|---|
| **Jev (TypeSafe)** | Hosted System One model. Questions are Noul, Choice, Score. Answers carry probabilities (and confidence where defined). |
| **judgevet** | Hex library around Jev: `SystemOnePort`, domain questions/answers, HTTP adapter, offline fakes, local policy. |
| **TypeLLM** | Open-weight path to Jev-like typed decisions: compile JSON Schema → decisions, score single-token choices, optional numeric FSM, optional permutation averaging (enum bias). Today wired to SGLang. |
| **typevet** | Hex rebuild of that open path on llama.cpp (and later peers), without SGLang. |

TypeLLM’s README states the inspiration explicitly and ships **JevBench**
evals under the published protocol (no permutation averaging). judgevet’s
docs stress the same trust split typevet must keep:
**valid structure ≠ correct judgment**; live transport proof ≠ calibration.

## What TypeLLM actually does

Portable core (see also closed research [#2](https://github.com/Alberto-Codes/typevet/issues/2)):

1. **`compile_json_schema`** → ordered `Decision` list (object root, properties,
   enums ≤ 24, booleans, bounded integer/number, open string and enum strings
   with `maxLength`, nullable via `["type","null"]`, `depends_on`, optional
   `return_probabilities` / permutations). `required` is validated at compile
   time; every property in the schema is still compiled and executed.
2. **Runtime `Choice`** binds labels; builds Field / Type / Instructions /
   Answer prompts; prefills `{"name":` for open or choice continuations.
3. **Categorical path** scores candidate token ids (logprobs), softmax,
   argmax or sample; optional **permutation averaging** reduces enum-order
   bias in the product — the published JevBench protocol does not use it.
4. **Numeric path** digit-by-digit FSM with tokenizer tables (not “dump a
   JSON number into a grammar and hope”).
5. **SGLang glue** (`sglang.py`) is the HTTP/logprob/prefix-cache adapter —
   replaceable. The decision engine is not.

Public call shape mirrors Jev’s vocabulary: exactly one of `state` or
`context`, and exactly one of `questions` or `schema` (mutual exclusion in
each pair), plus `model` → a mapping of field answers (with optional
probabilities).

## What judgevet needs from a local backend

judgevet callers already depend on:

- **Port:** `system_one(state, questions, model) → SystemOneResponse`
- **Questions:** `Noul`, `Choice`, `Score` with `instructions` (+ criteria)
- **Answers:** typed values plus **probability distributions** (Choice/Score;
  Noul is a yes-probability)
- **Hex discipline:** domain pure; outbound owns HTTP; fakes honor the same
  contract; verified vs inferred evidence ([architecture](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/architecture.md),
  [judgments](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/judgments.md),
  [verification](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/verification.md))

A future `HTTPSystemOneAdapter` peer could be a **typevet-backed** adapter
only if typevet can answer those primitives with distributions, not merely
emit a schema-valid JSON blob. No such judgevet adapter ships.

How typevet’s native questions map to decisions today:

| judgevet | typevet question → decision | Notes |
|---|---|---|
| Noul | `Noul` → Bool decision over `(False, True)` with probabilities | Noul has no separate confidence |
| Choice | `Choice` → Choice decision over criteria keys with probabilities | JSON Schema enums compile up to 24 choices. The compiler accepts a `permutations` budget, but execution rejects any value other than `1`; the JevBench protocol does not average permutations |
| Score | `Score` → Choice over rubric level indices `0..n-1` with probabilities | JSON Schema input still rejects `x-score`; use a closed number enum. Weighted mean is application/TypeSafe semantics |
| state | `JudgmentPort.judge(state, …)` | Same role: content under evaluation |

## What typevet has today

- `GenerationPort.generate(request)` → validated object via llama.cpp
  `response_format` / `json_schema`, with fail-fast schema validation after
  parse. Offline fake plus a live Gemma 4 proof on the local router.
- `compile_json_schema` and `execute_categorical_decision`: the TypeLLM
  compiler and categorical executor in `typevet.domain`. Neither performs I/O
  itself; the executor calls the injected scoring port.
- `CandidateScoringPort` and `LlamaCppCandidateScoringAdapter`: pre-sampling
  candidate logprobs from llama.cpp `/completion`.
- `JudgmentPort` with `Noul`, `Choice`, and `Score` questions, and
  `ScoringJudgmentAdapter`, which answers them from candidate scoring
  (judgevet-aligned vocabulary; no judgevet dependency).
- Evaluation harnesses: loader eval runner and TPJEP eight-task runner.

See [supported imports](../reference/supported-imports.md) for paths and
[native typed judgments](native-typed-judgments.md) for scope.

Live evidence is limited to recorded runs. The
[#133 finvet-derived receipt](../reference/judgment-live-receipts.md) records
probability-bearing Choice and Noul answers from a local llama.cpp router on
six rows plus two semantic controls. That is one small exploratory sample.
typevet does **not** yet prove:

- calibration, ECE, or task accuracy at scale
  ([#133](https://github.com/Alberto-Codes/typevet/issues/133) remains open)
- TypeLLM numeric FSM vs grammar-only numbers ([#12](https://github.com/Alberto-Codes/typevet/issues/12)); no numeric FSM ships
- native Gemma chat template parity ([#129](https://github.com/Alberto-Codes/typevet/issues/129))

Treat grammar-JSON as the **transport floor**. The **product spine** is the
decision runtime plus the consumer-facing `JudgmentPort`.

## Design rules so judgevet can use typevet later

1. Keep hex: domain types for requests/results stay IO-free; outbound owns
   llama.cpp (and later scoring).
2. Prefer **Decision / Choice** compilation in domain or a pure compiler
   module before freeform schema dump to the backend.
3. Keep `JudgmentPort` close enough to judgevet’s `SystemOnePort` shape that
   an adapter can wrap it — do not force judgevet to speak raw JSON Schema
   forever.
4. Preserve **probabilities** on categorical answers; grammar JSON alone
   usually drops them.
5. Keep judgevet’s evidence language: contract tests vs live; structure vs
   quality.
6. No SGLang dependency. Logprob and numeric paths must work on the
   llama.cpp (and peer) adapters.

## References

- Upstream: [TypeLLM/TypeLLM](https://github.com/TypeLLM/TypeLLM),
  [typellm.ai](https://typellm.ai/)
- Sister: [judgevet docs](https://github.com/Alberto-Codes/judgevet/tree/main/docs)
- typevet research: issues #2, #4, #11, #12; MVP epic #21
- Live judgment evidence: [judgment live receipts](../reference/judgment-live-receipts.md)
