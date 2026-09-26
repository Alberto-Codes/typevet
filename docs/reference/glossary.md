# Glossary

Kind: reference. This page gives the one meaning of each term that the typevet
pages use.

Every typevet page uses these words with these meanings only. The entries are
in alphabetical order.

## Terms

**acceptance review.** An independent check of a builder's diff against the
accepted issue contract. The reviewer changes no file.

**builder.** The worker role that implements one accepted contract inside named
paths and returns evidence. The builder never commits.

**Diátaxis.** Documentation standard at [diataxis.fr](https://diataxis.fr/).
Every page is one kind: tutorial, how-to, explanation or reference. Law in
this repo.

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

**supervisor.** The session that selects work, settles decisions, owns
acceptance and commits. The supervisor harness is independent of the worker
harness.

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
