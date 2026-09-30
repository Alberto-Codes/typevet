# Run the CORD expense smoke

Kind: how-to.

This page runs the bounded CORD expense triage smoke
([#165](https://github.com/Alberto-Codes/typevet/issues/165)). The smoke asks a
three-label question about 18 synthetic expense claims against six real CORD v2
receipts. It records semantic metrics and gates on structure and image
attachment only.

The smoke is not a production readiness claim, and it does not measure model
quality beyond these 18 claims.

## Before you start

- The llama.cpp router serves a Gemma 3 or Gemma 4 multimodal id with its
  vision projector. Common ids are `gemma-3-4b-it-q4km-mm` and
  `gemma-4-31b-kv9-q4km-mm`. See
  [Run the image-conditioned live smoke](run-a-multimodal-live-smoke.md).
- The alias `gemma-4-31b-kv9-q4km-mm` is not a Q4_K_M file. On the operator's
  router it loads `gemma-4-31b-24gib-kv9-decoder.gguf`, 16.0 GB.
  `/props` reports ftype `Q2_K - Medium`. The router uses a
  `--chat-template-file` override on image `server-cuda-b11243` ([#233](https://github.com/Alberto-Codes/typevet/issues/233)).
- `/apply-template` for that id renders the native Gemma 3 or Gemma 4 turn. The test
  fails on any other template family.
- The fixture is `tests/fixtures/cord/expense_smoke/` (CC-BY-4.0). Each receipt
  has one `supported`, one `contradicted` and one `insufficient` claim.

## Run the live smoke

```bash
TYPEVET_LLAMA__MULTIMODAL_MODEL=gemma-4-31b-kv9-q4km-mm \
  TYPEVET_LLAMA__TIMEOUT=900 \
  uv run pytest evals/tests/live/test_cord_expense_smoke_live.py -m live -q
```

Each pass writes an immutable receipt under
`scratchpad/cord-expense/receipt-<run_id>.json` (the prior
`receipt.json` name is not overwritten). A second write with the same
`run_id` fails and leaves the first file unchanged. Before any scoring call,
the harness snapshots prompt wording and three kinds of digests. The digests
cover the evaluated harness source, the expense-smoke manifest, and each receipt
PNG the smoke attaches. Finalization records call counts only and does not
re-read those paths. Thus late edits cannot change `experiment_identity` digests.

| Result | Cause |
|---|---|
| Skip | The router is down, or the model id is not in the catalog |
| Fail on `text-only input modalities` | The router serves that id without a projector |
| Fail on `the image was not attached` | The prompt token gap vs text or omission did not grow |
| Fail on a label or probability assertion | A response is not one of the three labels |
| Pass | Every request is typed and every image is attached |

A pass says nothing about accuracy. Read the metrics in the receipt.

## The three modalities

| Modality | Text | Image | Requests |
|---|---|---|---|
| `text_only` | The claim statement | None | 18 |
| `image_only` | A claim that states no amount | The receipt | 6 |
| `combined` | The claim statement | The receipt | 18 |

`image_only` runs once for each receipt, because its text is the same for all
three claims, plus one ``image_only`` omission control judgment. The total is
43 requests. The test asserts the total is 54 or
less.

## Read the receipt

- Each scored row may carry provenance from [#183](https://github.com/Alberto-Codes/typevet/issues/183):
  `application_mode`, `receipt_required`, `routing`, `model_calls`, and
  `deterministic_abstain`. When `combined` or `image_only` would run without
  receipt bytes, the harness returns `insufficient_evidence` with
  `routing=deterministic_missing_receipt` and **zero** model calls. That path is
  application validation, not model abstention, and must not inflate
  `#161` insufficient_abstention_rate as learned behavior.
- `metrics.combined` holds the semantic metrics: `accuracy`, per-verdict
  `recall`, `macro_recall`, `insufficient_abstention_rate`, `false_supported`
  and `confusion_gold_by_predicted`.
- `metrics.text_only` is the text prior. Without the receipt, the model cannot
  settle a claim, so use it as a baseline, not as a score.
- `image_only_insufficient_rate` is how often the model abstains when no
  amount is claimed.
- `soft_warnings` lists metrics below chance, an abstention rate under `0.5`,
  and claims routed to `supported` when gold says otherwise. The test does not
  fail on these warnings.

## Check the attachment yourself

Compare `tokens_evaluated` for the same claim in `text_only` and `combined`.
One Gemma 3 image costs 256 prompt tokens on the CORD smoke router.
A Gemma 4 image has no fixed cost.
Gemma 4 has a variable image-token budget that depends on the image.
In the #203 run, the gap was 228 to 1,108 tokens per receipt
([#203 per-claim table](https://github.com/Alberto-Codes/typevet/issues/203#issuecomment-5899231642)).
A silently dropped image grows the count by the marker text only.
The smoke counts that marker as 31 tokens.
The live harness calls `assert_cord_expense_live_smoke_gate` before scoring.
`typevet_evals.cord.expense_smoke` then sets one gap floor for each model.

| Model | Gap floor | Reason |
|---|---:|---|
| Gemma 3 | 225 tokens | Fixed 256-token image minus the 31-token marker |
| Gemma 4 | 197 tokens | Smallest measured 228-token image gap (#203) minus the 31-token marker |

Each arm is compared with its image-omitted control
([#260](https://github.com/Alberto-Codes/typevet/issues/260)).
For `image_only`, compare each receipt row to
`image_only.omission_tokens_evaluated`, the same claim text with no image.
The test passes the served family to `ScoringJudgmentAdapter` as
`served_template`. So `text_only` and `combined` share one native turn family,
and the gap holds the image and its marker only. Every `/completion` request
sends `"cache_prompt": false`.

## Historical run (Gemma 3)

On 2026-09-26 against `gemma-3-4b-it-q4km-mm`, recorded with the smoke from
revision `387285e`, before `aa1ad37`. At that revision the `text_only` prefix was
ChatML and the imaged prefix was the native Gemma 3 turn. So the token gap below
also holds a small template difference. Run the smoke again before you quote a
value for the current revision.

| Metric | `combined` | `text_only` |
|---|---|---|
| Accuracy | 0.333 | 0.278 |
| `supported` recall | 0.5 | 0.0 |
| `contradicted` recall | 0.5 | 0.833 |
| `insufficient` recall | 0.0 | 0.0 |
| `false_supported` | 4 | 1 |

The model did not choose `insufficient_evidence` in any modality, including
`image_only`. The image changed the label on 8 of 18 claims. Each attached
receipt added 256 prompt tokens once both arms used the native Gemma 3 turn at
`aa1ad37`.

## Historical run (Gemma 4, direct arm)

On 2026-09-26 against `gemma-4-31b-kv9-q4km-mm`, recorded with the smoke from
revision `25c5195`. The receipt does not record the file behind the alias.
On 2026-09-29 the alias loaded `gemma-4-31b-24gib-kv9-decoder.gguf`, ftype
`Q2_K - Medium` (not Q4_K_M), 16.0 GB. The router used a `--chat-template-file`
override on image `server-cuda-b11243` ([#233](https://github.com/Alberto-Codes/typevet/issues/233)). The vendored receipt is
`tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.json`. The router
declared vision and `/apply-template` rendered `native_gemma4_turn`.

The direct combined arm **passes** attachment and typed-label gates. It **fails**
two [#161](https://github.com/Alberto-Codes/typevet/issues/161) revision 1
semantic floors on the combined arm (recorded, not live-gated):

The vendored receipt’s `experiment_identity` block is flawed on this historical
run. `working_tree.dirty` was falsely `false`, and identity was captured after
scoring. The live harness path is missing from `code_path_digests`, and
`runtime.server_build` is `unknown`. See the sibling annotation
[`gemma4_kv9_direct_receipt.note.md`](https://github.com/Alberto-Codes/typevet/blob/main/tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.note.md).
Semantic FAIL rows below are unchanged; only identity metadata is annotated.

| #161 check | Limit | Measured | Status |
|---|---|---|---|
| Answerable accuracy | floor 0.67 | 0.25 | FAIL |
| Contradicted recall | floor 0.5 | 0.333 | FAIL |

Combined accuracy was 0.444. The model abstained on most imaged rows; that is a
quality read, not proof the receipt was unread.

## Check semantic acceptance on a saved receipt

After a live run writes a combined receipt JSON, evaluate the frozen #161
revision 1 floors offline. Exit code **0** means `accepted: true`; **1** means
checks failed; **2** means usage or malformed input.

```bash
uv run python scripts/check_cord_semantic_acceptance.py path/to/receipt.json
```

Equivalent module entry:

```bash
uv run python -m typevet_evals.cli.cord_semantic_acceptance path/to/receipt.json
```

Vendored examples (no live model):

```bash
uv run python scripts/check_cord_semantic_acceptance.py \
  tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.json

uv run python scripts/check_cord_semantic_acceptance.py \
  tests/fixtures/cord/semantic_acceptance/gemma4_post187_combined_pass.json
```

Pytest regression tests for the evaluator and CLI are separate from operator
acceptance; see [Release support matrix](../reference/typed-judgment-release-support-matrix.md).

## Known limits

- One model, one router build, six receipts, 18 claims.
- No calibration claim. Metrics are recorded, not gated.
- Claims are synthetic. Masked-digit claims stand in for unreadable evidence.
