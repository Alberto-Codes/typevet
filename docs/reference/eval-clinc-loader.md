# CLINC150 loader and domain-sharded Choice fixture

Kind: reference. Public CC BY 3.0 corpus for 15-way in-domain Choice per shard.
Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51);
research [#66](https://github.com/Alberto-Codes/typevet/issues/66) (accepted);
docs [#74](https://github.com/Alberto-Codes/typevet/issues/74); loader
[#73](https://github.com/Alberto-Codes/typevet/issues/73).

## Role in typevet

| Piece | Module / path |
|---|---|
| ``plus`` train loader | `typevet_evals.datasets.clinc` (`load_plus_split`) |
| Row mapping | `typevet_evals.datasets.clinc_rows` |
| Domain catalog and schemas | `typevet_evals.datasets.clinc_shard` |
| HF JSONL stream | `typevet_evals.datasets.clinc_download` |
| Domain intent map | `evals/src/typevet_evals/datasets/clinc_domains.json` (upstream `domains.json`) |
| Primary Choice | `intent` (15-enum per domain param) |
| Optional OOS Noul | `in_scope` (`yes` / `no`) when `include_oos=True` |
| Versioned Choice schema (banking default) | `evals/fixtures/clinc_banking_intent_choice_schema_v1.json` |
| Versioned OOS Noul schema | `evals/fixtures/clinc_in_scope_noul_schema_v1.json` |
| CI micro JSONL | `tests/fixtures/clinc/plus_banking_micro.jsonl` |

Shard geometry and **explicit non-mapping** to Banking77 /
`FRAUD_INTENTS`: [CLINC150 domain shard map](eval-clinc-shard-map.md).

## Labels and state

- Default ``domain`` param: **`banking`** (thematic default only; not a Banking77 label map).
- In-domain rows: gold ``choice_label`` ∈ 15 slugs for the selected domain.
- With ``include_oos=True``: in-domain rows also set ``noul_label=yes``; global
  ``oos`` intent rows set ``noul_label=no`` and omit Choice gold.
- Eval ``state`` is ``{"text": ...}`` only. Banned keys include ``intent``,
  ``domain``, ``expected``, and ``label``.

## Config

| HF config | Use in typevet v1 |
|---|---|
| `plus` | Default loader config (in-domain + optional OOS) |
| `small` | Out of scope for v1 loader |

## Related typevet pages

- [Banking77 proxy and metrics](banking77-proxy-and-metrics.md) — separate corpus and Noul proxy.
- [Complementary eval manifest](eval-complementary-manifest.md) — rank 5 slot.
