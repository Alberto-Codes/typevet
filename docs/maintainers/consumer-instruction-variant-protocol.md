# Frozen instruction-variant consumer protocol (revision 2)

Kind: reference, for maintainers. Parent [#177](https://github.com/Alberto-Codes/typevet/issues/177).

Post this table on #177 **before** any live rerun for slice 4. Bump
`instruction_variant_protocol_revision` when the table changes.

## Call budget (hard cap)

| Bucket | Count |
|---|---|
| Judgment calls (`ScoringJudgmentAdapter.judge`) | **4** (2 cases × 2 instruction variants, present image only) |
| Scoring requests (`score_candidates`) | **4** scheduled (**8** hard ceiling via ledger) |
| Metadata HTTP (factory props and template) | **2** |
| Tokenizer HTTP | **8** |
| Completion HTTP | **8** hard ceiling |

Each admitted attempt reserves a slot before dispatch. Failed calls consume slots;
refused reservations perform no IO. New receipts use
`dispatch_accounting_version: 1` with separate `attempts` and `successes`.
Factory capability metadata seeds the scoring cache. Offline negative checks
consume no HTTP slots. Historical revision 1 receipts remain unchanged.

## Case pins

| Row | `unique_data_id` | Gold (`true` / `false`) |
|---|---|---|
| C01 | `cmcc8u6yc00v91p1yw2eruz95` | true |
| C03 | `cmcc8u6yd00wv1p1yy8guorre` | false |

Fixture root: `tests/fixtures/psai/vision_smoke`.

## Instruction variants (caller-supplied, frozen for proof)

| Arm | Instruction text |
|---|---|
| seed | Is the website in this screenshot Fox News? |
| candidate | Does the attached screenshot show the Fox News website chrome (logo and header)? |

## Acceptance

| Leg | Expected |
|---|---|
| Variant matrix | Both arms run on both cases; rows and distributions retained |
| Invalid input | Empty model raises ``JudgmentValidationError``; failure retained |
| Unsupported template | ``JudgmentValidationError`` with ``served_template=None`` |
| Replay (#132) | ``compare_matched_prompt_outcomes`` descriptive v2 on saved distributions |
| Selection verdict | **None** — paired deltas on shared valid cases only; no promotion fields |

## Wheel + entry

```bash
uv run python scripts/run_consumer_instruction_variant_proof.py
TYPEVET_WHEEL_SHA256=$(sha256sum dist/typevet-*.whl | awk '{print $1}') \
  uv run python scripts/consumer_instruction_variant_proof.py
```

Live requires ``TYPEVET_REQUIRE_LIVE=1`` and an open llama.cpp router with Gemma 4 native vision.
