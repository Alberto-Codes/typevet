# typevet documentation

Kind: reference, the index. This page lists every typevet page and its kind.

The pages follow [Diátaxis](https://diataxis.fr/). Each page is one of four
kinds: a tutorial teaches, a how-to gives steps for a task, an explanation
answers a question, and a reference gives facts.

## Overview

- [README.md](../README.md) (explanation): what typevet is and where the
  worker harness lives.

## Explanation

- [TypeLLM, Jev and judgevet](explanation/typellm-and-judgevet.md): why the
  grammar-JSON MVP is a floor, what TypeLLM’s decision runtime preserves, and
  how judgevet could consume typevet later.
- [Verified evidence and inferred claims](explanation/verification.md): what
  unit, contract and live each prove; fixture labeling; link to the testing
  pyramid law.
- [Library-first architecture](explanation/library-first-architecture.md): why
  the importable library is the artifact; composition root vs explicit adapter
  args; hex layers and judgevet alignment.

## How-to

- [Run Gemma 4 on llama.cpp](how-to/run-gemma4-llamacpp.md): local router,
  preset, and live pytest for the MVP path.
- [Call typevet from Python](how-to/call-typevet-from-python.md): Fake and
  llama.cpp adapters, `generate()` vs port injection; link to Gemma howto.

## Architecture decisions (planned)

`docs/adr/` is not present yet. When hex and related judgments lock, add an
ADR log in the automarket shape (`NNNN-slug.md`). Tracked on the “adopt ADR
log” epic. Until then, accepted judgment comments on issues are the record.

## Reference

- [Eval partner data policy](reference/eval-partner-data-policy.md) (reference):
  public eval datasets vs collections NBA partner exclusion and CI markers.
- [Testing pyramid](reference/testing.md) (reference): unit / contract / live markers and what each layer proves.
- [Errors](reference/errors.md)
 (reference): domain vs adapter generation errors and llama.cpp mapping.
- [Configuration](reference/configuration.md) (reference): `TYPEVET_LLAMA__*`
  composition-root settings and future CLI hookup; links to log settings.
- [Diagnostic events](reference/diagnostic-events.md) (reference): stderr
  structlog closed-set events for generation and HTTP (`http.request`,
  `generation.call`).
- [CFPB and synth seed-only](reference/eval-cfpb-synth-seed-only.md)
  (reference): no public gold intent; synthetic contract fixture labeling.
- [Banking77 proxy and metrics](reference/banking77-proxy-and-metrics.md) (reference): six-intent proxy; Noul vs Choice metric claims.
- [go_emotions conversion and prune](reference/go-emotions-conversion-and-prune.md) (reference): strict exactly-one gold, 24-enum prune, neutral co-label rule.
- [CLINC150 domain shard map](reference/eval-clinc-shard-map.md) (reference): 10×15 Choice shards; CC BY 3.0; no Banking77 / FRAUD_INTENTS mapping.
- [CLINC150 loader and domain Choice](reference/eval-clinc-loader.md) (reference): `plus` config; default `banking` shard; optional OOS Noul.
- [DIFrauD loader and is_scam fixture](reference/eval-difraud-loader.md) (reference): SMS-default domain flag, natural binary Noul, class imbalance notes.
- [Civil Comments loader and is_toxic fixture](reference/eval-civil-comments-loader.md) (reference): τ=0.5 toxicity Noul, balanced tiers A/B, sensitive-text policy.
- [go_emotions loader and emotion Choice fixture](reference/eval-go-emotions-loader.md) (reference): simplified strict exactly-one; 24-enum; Apache 2.0.
- [PubMedQA loader and answer Choice fixture](reference/eval-pubmedqa-loader.md) (reference): pqa_labeled yes/no/maybe; contexts state; MIT.
- [BoolQ loader and answer Noul fixture](reference/eval-boolq-loader.md) (reference): validation split, no/yes Noul, CC BY-SA smoke cap, HF bulk stream.
- [Hyperpartisan loader and hyperpartisan Noul fixture](reference/eval-hyperpartisan-loader.md) (reference): byarticle train only, HTML cleanup, stratified holdout, excludes bypublisher.
- [Complementary eval manifest](reference/eval-complementary-manifest.md) (reference): JevBench-primary ranked open sets; see also `evals/`.
- [Glossary](reference/glossary.md) (reference): the one meaning of each term.
- [Supported imports](reference/supported-imports.md) (reference): package
  `__all__` surfaces and `from typevet…` paths for library callers.
- [Worker runs](reference/worker-runs.md) (reference): the launch evidence and
  commit trailers for each worker.
- [Writing system](reference/writing-system.md) (reference): Diátaxis,
  ASD-STE100 local profile and prose modes.
- [Commit messages](reference/commits.md) (reference): Conventional Commits
  1.0.0 vocabulary for this repo.

## For maintainers and agents

- [CLAUDE.md](../CLAUDE.md) (reference and how-to, for agents): start-here for
  fresh sessions, issue bus, non-negotiables and commit rules.
- [Groom worker issues](maintainers/groom-worker-issues.md) (how-to): file,
  triage, size and split work on GitHub before delegation.
- [Delegate a bounded change](maintainers/delegate-work.md) (how-to): brief,
  harness, model weight and acceptance after a contract exists.
- [Verify package typing and version](maintainers/verify-package.md) (how-to):
  wheel `py.typed` marker, isolated consumer, and `__version__` checks before
  publish.
