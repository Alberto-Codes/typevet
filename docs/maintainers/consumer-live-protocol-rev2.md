# Frozen live protocol — durable harness rev 2

Kind: reference, for maintainers. Parent [#177](https://github.com/Alberto-Codes/typevet/issues/177).

Post this table on #177 **before** any live rerun. Bump `consumer_live_protocol_revision` when the table changes.

## Call budget (hard cap)

| Bucket | Count |
|---|---|
| Judgment calls (`ScoringJudgmentAdapter.judge`) | **14** (12 visual present/omit/swap + 2 text annotation) |
| Scoring requests (`score_candidates`) | **16** (12 visual Noul + 4 annotation Choice/Noul) |
| Auxiliary HTTP (health + capability + template probe) | **3** |

## Case pins (`FROZEN_CONSUMER_CASE_UIDS`)

| ID | Row | `shows_fox_news_chrome` gold | Annotation leg |
|---|---|---|---|
| C01 | `cmcc8u6yc00v91p1yw2eruz95` | true | yes (Choice + Noul) |
| C03 | `cmcc8u6yd00wv1p1yy8guorre` | false | no |
| C08 | `cmcc8u6ym018l1p1yxhf18gc2` | false | yes (Choice + Noul) |
| C10 | `cmcc8u6yc00va1p1ydsdu52zy` | true | no |

Fixture root: `tests/fixtures/psai/vision_smoke` (committed manifest + screenshots; no scratchpad graft).

## Expected outcomes (acceptance)

| Leg | Expected |
|---|---|
| Visual present/swapped | Noul polarity matches row gold; paired Fox image outscores non-Fox on each row |
| Visual omitted | Never credited as a semantic hit |
| Annotation (C01, C08) | `category` Choice equals gold; `requires_login` Noul polarity equals gold |
| Unsupported template | `JudgmentValidationError` with `served_template=None` (no scoring HTTP) |
| Template class | `NATIVE_GEMMA4_TURN` from `/apply-template` on the live model id |

## Wheel + receipt

- Build wheel from proof tip; record measured sha256 on receipt and `proof-manifest-rev2.json`.
- Run offline + optional live via `uv run python scripts/run_psai_vision_consumer_proof.py` (isolated `--with` wheel).
- Write live receipts only under `tests/fixtures/consumer/` using exclusive paths (never overwrite `live-receipt-v1-checkout.json`).
