# CFPB and synth collections — seed-only (no public gold intent)

Kind: reference. This page states how typevet may use CFPB narratives and
finvet’s synthetic collections intent data in public eval work.

Parent: [#51](https://github.com/Alberto-Codes/typevet/issues/51). Baseline:
[#53](https://github.com/Alberto-Codes/typevet/issues/53) research return.
Public eval direction: [eval partner data policy](eval-partner-data-policy.md).

## What finvet uses them for

**CFPB Consumer Complaint Database** (US public record, debt-collection
product filter) supplies **unlabelled** complaint narratives. finvet reads a
local CSV export. The loader does not attach collections intent labels.

**Synthetic collections intent** data is **derived**. finvet builds it from
CFPB seeds plus a rewriter and Jev filtering. It is not a fixed public
corpus with stable gold labels unless a separate regen or publish spec
accepts one.

## typevet policy (v1)

typevet treats CFPB and synth collections as **seed-only** for public eval
claims:

| Path | Allowed in typevet public artifacts | Allowed eval claim |
|---|---|---|
| CFPB CSV (local, operator-owned) | No checked-in CFPB bulk in this repo | **No** “CFPB eval” or gold-intent benchmark |
| finvet synth collections output | Only if a later issue accepts fixtures | **No** public gold intent without regen/publish spec |
| Optional contract fixtures from synth | Yes, when an eval issue adds them | Must follow fixture labeling below |

Adopt **Banking77** and **DIFrauD** for shippable public eval direction per
#53. Do not rank CFPB or synth beside them as labeled benchmarks.

## Fixture labeling

When typevet adds **optional** contract fixtures built from synth (or from
CFPB-shaped text), label them as **synthetic fixtures** per
[verified evidence and inferred claims](../explanation/verification.md#fixture-labeling-judgevet-shape).
State in the fixture metadata or test docstring that the text is synthetic
and that no gold collections intent is claimed from CFPB alone.

A green contract test on such a fixture verifies adapter wiring for that
case. It does not verify model judgment quality or a public leaderboard score.

## Claims to avoid

Do not write or imply:

- “CFPB eval”, “CFPB benchmark”, or “gold intent from CFPB”.
- A frozen synth corpus as a public dataset without an accepted publish spec.
- Agreement, ECE, or calibration claims on synth-derived labels before #50 and
  the Noul/Choice logprob path ([#11](https://github.com/Alberto-Codes/typevet/issues/11),
  [#22](https://github.com/Alberto-Codes/typevet/issues/22)).

## When this changes

A child issue under #51 may accept a **regen spec** (how to rebuild synth) or a
**publish spec** (how to freeze and name a public fixture set). Until then,
seed-only stands. This page does not document the finvet synthesis pipeline;
see finvet `docs/reference/datasets.md` for that workflow.
