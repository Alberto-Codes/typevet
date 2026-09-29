# Run the PSAI vision smoke

Kind: how-to.

This page runs the bounded PSAI image+question smoke
([#154](https://github.com/Alberto-Codes/typevet/issues/154)). The smoke asks
typed questions about five real computer-use screenshots and proves the pixels,
not the text, move the visually dependent answer.

The smoke reports typed-answer validity and matched image controls. It does not
measure model quality, and it does not claim the model succeeds at GUI tasks.

## What the fixture holds

`tests/fixtures/psai/vision_smoke/` holds five unmodified first screenshots from
`anaisleila/computer-use-data-psai` (MIT), one PNG per `unique_data_id`, plus
`manifest.json`.

| `unique_data_id` | Visual family | First screenshot |
|---|---|---|
| `cmcc8u6yc00va1p1ydsdu52zy` | `fox_news` | foxnews.com privacy policy |
| `cmcc8u6yc00v91p1yw2eruz95` | `fox_news` | Fox News Media corporate |
| `cmcc8u6yc00vm1p1yhjl1u0bf` | `fox_news` | foxcareers.com job search |
| `cmcc8u6yd00wv1p1yy8guorre` | `non_fox` | Home Depot pendant lights |
| `cmcc8u6yd00wr1p1yj7aot3ae` | `non_fox` | Home Depot store finder |

The manifest keeps `dataset_id`, `license`, `split`, `stream_order`,
`screenshot_index` and a `sha256` per PNG. Every gold answer lives under
`expected` and nowhere else.

## The questions

Two gold provenances sit side by side. `gold_provenance` in the manifest records
which is which.

| Question | Type | Gold provenance |
|---|---|---|
| `category` | Choice | `annotation` — Hub `category` |
| `requires_login` | Noul | `annotation` — Hub `requires_login` |
| `shows_fox_news_chrome` | Noul | `manual_visual` — hand labelled |

No Hub field describes the pixels, so the visually dependent question is labelled
manual. Do not read it as corpus gold.

## Why the visual leg drops `task_name`

Every PSAI `task_name` pins its site, for example `Only use
http://foxnews.com to achieve this task`. That text hands the model the
`shows_fox_news_chrome` answer, so a state built from `task_name` would let the
model pass without ever looking at the image.

The visual leg therefore uses `VISUAL_CONTROL_STATE`, a family-neutral sentence.
`validate_no_family_leak` rejects any visual state that names a family, and a
unit test asserts every `task_name` fails that check. The annotation leg keeps
the `{task_name}` state, because the Hub labels are not recoverable from the
text as literal strings.

## Run the offline tests

```bash
uv run pytest evals/tests/unit/test_psai_vision_fixtures.py \
  evals/tests/unit/test_psai_vision_controls.py \
  evals/tests/contract/test_psai_vision_smoke_contract.py -q
```

These check fixture digests, gold placement, the leakage ban and the control
matrix shape against a fake scorer. They need no network and no model.

## Run the live smoke

```bash
TYPEVET_LLAMA__MULTIMODAL_MODEL=gemma-3-4b-it-q4km-mm \
  TYPEVET_LLAMA__TIMEOUT=900 \
  uv run pytest evals/tests/live/test_psai_vision_smoke_live.py -m live -q
```

The smoke writes a receipt to `scratchpad/psai-vision/live_receipt.json`. See
[Run the image-conditioned live smoke](run-a-multimodal-live-smoke.md) for the
router requirements, the nested prompt shape and the complete library recipe.

The smoke renders one turn through `POST /apply-template`, classifies it with
`classify_served_template`, and passes the family to `ScoringJudgmentAdapter`
as `served_template`. Media scoring needs `native_gemma3_turn`. With that
family, all three controls use the same Gemma 3 turn wrapper, so the `omitted`
prefix differs from the imaged prefixes only in the media marker. Every
`/completion` request sends `"cache_prompt": false`.

| Result | Cause |
|---|---|
| Skip | The router is down, or the model id is not in the catalog |
| Fail on `text-only input modalities` | The router serves that id without a projector |
| `JudgmentValidationError` on `served template` | `/apply-template` renders a family other than native Gemma 3 |
| Fail on `the image was not attached` | The prompt token count did not grow |
| Fail on `did not move the judgment` | Swapping the screenshot changed nothing |
| Pass | Each row's own screenshot outscored the opposite family's |

## The three controls

For every row the smoke runs the same question three ways.

| Condition | Image | Counts as a hit |
|---|---|---|
| `present` | The row's own screenshot | yes |
| `omitted` | None | **never** |
| `swapped` | A screenshot from the other family | yes |

All three carry byte-identical text. An omitted image is recorded and never
credited: with no image the model still answers from its prior, so a match there
measures the prior rather than the pixels.

## Historical run

Every measured value on the rest of this page comes from one receipt recorded at
revision `3ecea25` with `gemma-3-4b-it-q4km-mm`. That revision predates
`cache_prompt: false` (`5c5df49`) and native Gemma 3 turns (`9d8d818`,
`aa1ad37`). Run the smoke again before you quote a value for the current
revision.

On that run every omitted row returned the same `0.7545`, which is exactly what
a blind harness would return for all fifteen rows.

## How the gate reads the result

`paired_image_ordering` is the gate. For each row it compares the two imaged
controls against each other. One carries a Fox screenshot and one carries a
non-Fox screenshot, under identical text, so the Fox one must score higher by
`PAIRED_MARGIN_FLOOR`.

A paired comparison needs no calibration assumption. Reading one probability
against `NOUL_THRESHOLD` does, and this model did not meet it on the historical
run at `3ecea25`: the Home Depot store finder scored `0.6929` on "Is the
website in this screenshot Fox News?", above the `0.5` cut, while every Fox
screenshot scored `0.90` or higher. The
smoke records that threshold reading under `threshold_diagnostic` and does not
gate on it.

## Check the attachment yourself

A valid distribution is not proof that the image arrived. Read
`tokens_evaluated` in the receipt. One Gemma 3 image costs 256 prompt tokens.
The live test asserts a gap of at least 200 tokens over the omitted baseline
for every imaged control. The historical run at `3ecea25` shows `363` against
`104`.

## Known limits

- One model, one router build, five screenshots. No corpus claim.
- No calibration claim. The smoke reports probabilities, not reliability.
- `shows_fox_news_chrome` is hand labelled, so it carries author judgement.
- The adapter sends `"cache_prompt": false` since `5c5df49`. The historical run
  at `3ecea25` used the router default, so its values can differ slightly from a
  run at the current revision.
- One served template family, `native_gemma3_turn`. The smoke makes no claim
  about any other model family.
