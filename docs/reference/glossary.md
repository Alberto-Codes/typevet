# Glossary

Kind: reference. This page gives the one meaning of each term that the typevet
pages use.

Every typevet page uses these words with these meanings only. The entries are
in alphabetical order.

## Terms

**acceptance review.** An independent check of a builder's diff against the
accepted issue contract. The reviewer changes no file.

**Banking77 proxy label.** In typevet/finvet evals, PolyAI Banking77 intents
collapsed via finvet’s six `FRAUD_INTENTS` into binary fraud / not_fraud for
Noul-primary metrics. See
[Banking77 proxy and metrics](banking77-proxy-and-metrics.md).

**CLINC domain shard.** One of ten topical domains in CLINC150 OOS eval
(`domains.json`). Each shard exposes exactly **15** intent slugs as a single
**Choice** task (≤24 labels). Shards do **not** map to Banking77 categories or
finvet **`FRAUD_INTENTS`**. See
[CLINC150 domain shard map](eval-clinc-shard-map.md).

**composition root.** The process or module that reads configuration,
constructs adapters, runs work, and closes adapters when work ends. A future
typevet CLI or MCP inbound adapter owns this role. Direct library callers
own it in application code.

**Decision.** A TypeLLM-compiled field from JSON Schema (enum, boolean,
bounded number or open string). The portable core executes Decisions; it is
not the same as dumping a whole object through a grammar.

**generation port.** typevet’s MVP `GenerationPort`: prompt + JSON Schema +
model → validated object. The transport floor on llama.cpp.

**judgment port.** typevet’s IO-free `JudgmentPort`: `state` + named
`Noul` / `Choice` / `Score` questions + model → typed answers in a
`JudgmentResponse`. Vocabulary aligns with judgevet `SystemOnePort`; no
judgevet import. Logprob scoring adapters remain separate work ([#26](https://github.com/Alberto-Codes/typevet/issues/26)).

**Jev.** TypeSafe’s hosted System One model. judgevet calls it over HTTP.
TypeLLM and typevet target open-weight typed decisions in the same family.

**judgevet.** Sister hex client for Jev. Port shape: `system_one(state,
questions, model)` with Noul / Choice / Score answers and probabilities. A
future consumer of typevet if typevet exposes a compatible judgment surface.

**System One.** TypeSafe’s typed-judgment model family (Noul, Choice, Score).
TypeLLM is explicitly inspired by it. typevet aims at a local open path to
similar guarantees.

**Copilot.** GitHub Copilot CLI (`copilot`). Present on this operator
machine. For typevet worker dispatches, request `--model auto` only. Named
Copilot models (including Fable) are out.

**flat module.** One concern in one `.py` file at the package level. Prefer
`domain/models.py` over a nested `domain/models/` package. Package
`__init__.py` files re-export the public surface and carry full module docs
so agents and callers can discover names from the init.

**glossary.** The one allowed meaning of each domain term. One term per
concept. The writing system binds prose to it.

**harness.** The program that runs a model and gives it tools. Orthogonal to
role and to model weight. typevet verifies three worker harnesses: pi, the
Claude Code Agent tool and the Cursor CLI. Supervisors may also use Codex,
Copilot or others.

**issue bus.** GitHub issues as the durable record for asks, triage,
contracts, handoffs and acceptance across sessions and harnesses. Chat is not
the bus.

**judgment.** An issue label: a decision remains; the supervising or
orchestrating model must settle it before `worker-fit` implementation.

**model weight.** Capacity class for a dispatch: `light`, `medium` or `heavy`.
Orthogonal to harness. A light model may run on Claude, Cursor or pi; a heavy
model likewise. Do not equate weight with a brand name.

**reviewer.** The role that exercises the defining behaviour and reports
findings. A model name alone does not make a review independent.

**specifier.** The role that writes a definition of ready and done for one
issue. The specifier never edits code.

**testing pyramid.** Three layers: `unit` (fast, no network), `contract`
(fake and real adapter share fixtures), `live` (opt-in exercised call).
Default pytest excludes `live`. Coverage ≥ 90 on the default suite. Law in
`CLAUDE.md`. Sisters: judgevet markers; gepa-adk ADR-005.

**uv.** Astral’s package and tool runner. Ground floor for typevet: sync,
locks, `uv run` for every gate and script. No parallel pip/poetry install path.

**worker.** A bounded session that follows a brief, edits only allowed paths
(or returns a named research artifact) and returns evidence. A worker never
commits unless the brief assigns that action.

**worker-fit.** An issue label: the contract is decided and mechanical enough
for any verified worker harness. It does not mean “use pi” and does not pick
model weight.

**worker trailer.** A `Generated-By` or `Specified-By` commit trailer that
records which worker model produced code or a specification. It is evaluation
evidence, not authorship. Never use `Co-Authored-By` for a model.

**writing system.** The local prose profile informed by ASD-STE100 Issue 9,
plus Diátaxis page kinds and Conventional Commits. See
[writing-system.md](writing-system.md). Not a claim of full ASD-STE100
compliance.
