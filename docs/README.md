# typevet documentation

Kind: reference, the index. This page lists every typevet page and its kind.

The pages follow [Diátaxis](https://diataxis.fr/). Each page is one of four
kinds. A tutorial teaches, and a how-to gives steps for a task. An explanation
answers a question, and a reference gives facts.

## Overview

- [README.md](https://github.com/Alberto-Codes/typevet/blob/main/README.md) (explanation): what typevet is, its status
  and a quickstart.
- [Landing page](index.md) (reference): the site home page, with a path for
  each reader question.

## Explanation

- [How typevet works with Gemma 4](explanation/how-typevet-works-with-gemma-4.md):
  typed judgment and schema-bound generation on llama.cpp and vLLM, with receipts.
- [TypeLLM, Jev and judgevet](explanation/typellm-and-judgevet.md): why
  grammar-JSON is the transport floor. It names which TypeLLM decision ideas typevet
  ships (compiler, candidate scoring, `JudgmentPort`) and what stays unproven.
- [Verified evidence and inferred claims](explanation/verification.md): what
  unit, contract and live each prove; fixture labeling; link to the testing
  pyramid law.
- [Library-first architecture](explanation/library-first-architecture.md): why
  the importable library is the artifact. It covers composition root vs explicit adapter
  args; hex layers including runtime and evaluation; module command vs console
  script.

- [Native typed judgments](explanation/native-typed-judgments.md): what typevet
  owns after M2 — questions → Gemma judgments → TPJEP eight-task smoke; validity
  vs calibration; control-token binding.
- [Limits and known gaps](explanation/limits.md): the known limits and open gaps, each with its source, and the page status values.
- [Gemma 4 multimodal judgments](explanation/gemma-4-multimodal-judgments.md): how images reach Gemma 4 on llama.cpp and vLLM, what the receipts prove, and the limits.
- [Two-image face matching](explanation/two-image-face-matching.md): what the LFW face-match runs measured and what the probability means. The results are not an identity check.
- [Two-image signature comparison](explanation/two-image-signature-comparison.md): what the CEDAR signature-match runs measured and what the probability means. The results are not a fraud control.
- [Check images against a synthetic register](explanation/check-register-matching.md): what the generated check-match runs measured, the false-clear rate and a legibility gate hypothesis. The results are not a fraud control.

## Tutorials

- [First typed judgment offline](tutorials/first-typed-judgment-offline.md): Fake outbound adapter through JudgmentPort; no network.
- [First typed judgment on llama.cpp](tutorials/first-typed-judgment-on-llama-cpp.md): the same three questions against a local Gemma 4 llama.cpp server.

## How-to

- [Install typevet](how-to/install.md): install the PyPI release with
  `pip install typevet` or `uv add typevet`, or build a wheel from a checkout.
- [Run Gemma 4 on llama.cpp](how-to/run-gemma4-llamacpp.md): stock
  `llama-server`, nested `json_schema`, and opt-in live pytest.
- [Serve typevet on vLLM](how-to/serve-typevet-on-vllm.md): the tested
  vLLM v0.30.0 pin, `TYPEVET_VLLM__*` settings and the live acceptance test (#170).
- [Serve Gemma 4 31B on a rented H100](how-to/serve-gemma-4-31b-on-a-rented-h100.md):
  one RunPod H100, stock vLLM, one judgment (#236).
- [Call typevet from Python](how-to/call-typevet-from-python.md): Fake and
  llama.cpp adapters, `generate()` vs port injection; link to Gemma howto.

- [Run a small live judgment eval](how-to/run-a-small-live-judgment-eval.md):
  frozen six-message finvet workload, semantic controls, receipt paths (#133).

- [Run the image-conditioned live smoke](how-to/run-a-multimodal-live-smoke.md):
  opt-in vision check, `media=` calls, and the llama.cpp nested prompt shape.

- [Run the CORD expense smoke](how-to/run-the-cord-expense-smoke.md) (how-to): 18 synthetic claims on six CORD receipts; semantic metrics + attachment gates ([#165](https://github.com/Alberto-Codes/typevet/issues/165)).
- [Run the PSAI vision smoke](how-to/run-the-psai-vision-smoke.md): five
  vendored computer-use screenshots with annotation-backed and manual visual
  questions. It has matched present / omitted / swapped image controls (#154).
- [Connect Gemma 4 native vision judgment](how-to/connect-gemma4-native-vision-judgment.md):
  one `typevet.runtime` factory composes the llama.cpp scoring adapter and
  `ScoringJudgmentAdapter` for Gemma 4 native-turn vision.
- [Use typevet as a judgevet provider](how-to/use-typevet-as-a-judgevet-provider.md):
  the `typevet[judgevet]` extra, a judgevet `SystemOnePort` over typevet, and the
  declared capabilities (#284).

## Architecture decisions

- [ADR index](adr/README.md) (reference): accepted architecture decisions and their source evidence.
- [0001: Runtime orchestration](adr/0001-runtime-orchestration.md) (reference): thin runtime facades without an engine layer.
- [0002: Package layout](adr/0002-package-layout.md) (reference): the narrow
  library root and the `evals/` workspace member.

## Reference

- [Eval partner data policy](reference/eval-partner-data-policy.md) (reference):
  public eval datasets vs collections NBA partner exclusion and CI markers.
- [Testing pyramid](reference/testing.md) (reference): unit / contract / live markers and what each layer proves.
- [Errors](reference/errors.md)
 (reference): domain vs adapter generation errors and llama.cpp mapping.
- [Configuration](reference/configuration.md) (reference): `TYPEVET_LLAMA__*`
  composition-root settings and future CLI hookup; links to log settings.
- [Security](reference/security.md) (reference): what typevet sends where, API key masking, redaction and vulnerability reports.
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
- [Live eval runner](reference/eval-live-runner.md) (reference): opt-in Banking77/BoolQ slice via GenerationPort; attempted / schema-valid / gold-match counts.
- [Judgment live receipts](reference/judgment-live-receipts.md) (reference): measured #133 live predictions, latencies, semantic controls, known limitations.
- [Performance on one H100](reference/performance.md) (reference): vLLM
  throughput per concurrency level, calibration per set, cold start, cost and limits (#236).
- [Hyperpartisan loader and hyperpartisan Noul fixture](reference/eval-hyperpartisan-loader.md) (reference): byarticle train only, HTML cleanup, stratified holdout, excludes bypublisher.
- [CEDAR signature loader and signature-match request](reference/eval-cedar-loader.md) (reference): pinned CEDAR archive, balanced 60 + 60 + 60 slice, two-image judgment, no signature bytes stored.
- [LFW View 2 loader and face-match request](reference/eval-lfw-loader.md) (reference): pinned figshare files, balanced 100 + 100 slice, two-image judgment, no face bytes stored.
- [Synthetic checks and the check-match run](reference/eval-synthetic-checks.md) (reference): seeded 20 × 7 generated checks, one-image judgment, metrics, key-free receipt, no renders stored.
- [Complementary eval manifest](reference/eval-complementary-manifest.md) (reference): JevBench-primary ranked open sets; see also `evals/`.
- [TPJEP eight-task runner](reference/eval-tpjep-runner.md) (reference):
  eight vendored JevBench rows as native `Noul` / `Choice` / `Score` questions
  through `JudgmentPort`.
- [TPJEP attempt result records](reference/eval-tpjep-records.md) (reference):
  one frozen JSONL object per scheduled attempt, required fields and summary.
- [PSAI metadata loader and Decision map](reference/eval-psai-metadata-map.md)
  (reference): metadata-only computer-use PSAI slice, field map and license.
- [PSAI vision Choice probability evidence](reference/psai-vision-choice-probability-evidence.md)
  (reference): offline raw probability fixtures and the completeness verdict
  ([#180](https://github.com/Alberto-Codes/typevet/issues/180)).
- [Question records → JSON Schema](reference/question-schema-map.md) (reference): Noul/Choice/Score export records to `compile_json_schema` fixtures ([#102](https://github.com/Alberto-Codes/typevet/issues/102)).
- [Glossary](reference/glossary.md) (reference): the one meaning of each term.
- [Python API reference](reference/api/index.md) (reference): the index of
  the package pages generated from the public docstrings. The pages are
  [root package](reference/api/root.md), [domain](reference/api/domain.md),
  [ports](reference/api/ports.md), [runtime](reference/api/runtime.md) and
  [inbound adapters](reference/api/inbound.md).
- [Supported imports](reference/supported-imports.md) (reference): package
  `__all__` surfaces, modules removed before 0.1.0, and the module command entry.
- [Releases](reference/releases.md) (reference): PyPI and GitHub release channels, release-please version cuts and pre-1.0 stability.
- [Typed-judgment release support matrix](reference/typed-judgment-release-support-matrix.md)
  (reference): supported sync APIs, Gemma 4 native vision pins, evidence
  commands, and release exclusions ([#191](https://github.com/Alberto-Codes/typevet/issues/191)).
- [Worker runs](https://github.com/Alberto-Codes/typevet/blob/main/docs/reference/worker-runs.md) (reference): the launch evidence and
  commit trailers for each worker.
- [Writing system](reference/writing-system.md) (reference): Diátaxis,
  ASD-STE100 local profile and prose modes.
- [Commit messages](reference/commits.md) (reference): Conventional Commits
  1.0.0 vocabulary for this repo.
- [LOC gate](reference/loc-gate.md) (reference): what the line-of-code gate
  counts and its limits.

## For maintainers and agents

- [CLAUDE.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md) (reference and how-to, for agents): start-here for
  fresh sessions, issue bus, non-negotiables and commit rules.
- [Groom worker issues](https://github.com/Alberto-Codes/typevet/blob/main/docs/maintainers/groom-worker-issues.md) (how-to): file,
  triage, size and split work on GitHub before delegation.
- [Delegate a bounded change](https://github.com/Alberto-Codes/typevet/blob/main/docs/maintainers/delegate-work.md) (how-to): brief,
  harness, model weight and acceptance after a contract exists.
- [Verify package typing and version](https://github.com/Alberto-Codes/typevet/blob/main/docs/maintainers/verify-package.md) (how-to):
  wheel `py.typed` marker, isolated consumer, and `__version__` checks before
  publish.
- [Issue #177 handoff](https://github.com/Alberto-Codes/typevet/blob/main/docs/maintainers/issue-177-handoff.md) (reference): phase status,
  freeze protocol and commands for the consumer proof.
- [Frozen live protocol, rev 2](https://github.com/Alberto-Codes/typevet/blob/main/docs/maintainers/consumer-live-protocol-rev2.md) (reference):
  call budget, case pins and expected outcomes for a #177 live rerun.
- [Frozen instruction-variant protocol](https://github.com/Alberto-Codes/typevet/blob/main/docs/maintainers/consumer-instruction-variant-protocol.md)
  (reference): call budget, case pins and instruction variants for #177 slice 4.
