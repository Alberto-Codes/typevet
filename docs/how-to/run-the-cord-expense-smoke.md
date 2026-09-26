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

- The llama.cpp router serves `gemma-3-4b-it-q4km-mm` with its vision
  projector. See
  [Run the image-conditioned live smoke](run-a-multimodal-live-smoke.md).
- `/apply-template` for that id renders the native Gemma 3 turn. The test
  fails on any other template family.
- The fixture is `tests/fixtures/cord/expense_smoke/` (CC-BY-4.0). Each receipt
  has one `supported`, one `contradicted` and one `insufficient` claim.

## Run the live smoke

```bash
TYPEVET_LLAMA__MULTIMODAL_MODEL=gemma-3-4b-it-q4km-mm \
  TYPEVET_LLAMA__TIMEOUT=900 \
  uv run pytest tests/live/test_cord_expense_smoke_live.py -m live -q
```

The smoke writes `scratchpad/cord-expense/receipt.json`.

| Result | Cause |
|---|---|
| Skip | The router is down, or the model id is not in the catalog |
| Fail on `text-only input modalities` | The router serves that id without a projector |
| Fail on `the image was not attached` | The prompt token count did not grow |
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
three claims. The total is 42 requests. The test asserts the total is 54 or
less.

## Read the receipt

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
One Gemma 3 image costs 256 prompt tokens. The test asserts a gap of at least
200. The test passes the served family to `ScoringJudgmentAdapter` as
`served_template`, so `text_only` and `combined` both use the native Gemma 3
turn. The gap holds the image and its marker, with no template difference.
Every `/completion` request sends `"cache_prompt": false`.

## Historical run

On 2026-09-26 against `gemma-3-4b-it-q4km-mm`, recorded with the smoke from
revision `387285e`, before `aa1ad37`. At that revision the `text_only` prefix was ChatML and the imaged
prefix was the native Gemma 3 turn, so the token gap below also holds a small
template difference. Run the smoke again before you quote a value for the
current revision.

| Metric | `combined` | `text_only` |
|---|---|---|
| Accuracy | 0.333 | 0.333 |
| `supported` recall | 0.5 | 1.0 |
| `contradicted` recall | 0.5 | 0.0 |
| `insufficient` recall | 0.0 | 0.0 |
| `false_supported` | 4 | 10 |

The model did not choose `insufficient_evidence` in any modality, including
`image_only`. The image changed the label on 9 of 18 claims. Each attached
receipt added 245 prompt tokens.

## Known limits

- One model, one router build, six receipts, 18 claims.
- No calibration claim. Metrics are recorded, not gated.
- Claims are synthetic. Masked-digit claims stand in for unreadable evidence.
