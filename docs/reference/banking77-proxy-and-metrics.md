# Banking77 proxy label and Choice vs Noul metrics

Kind: reference. This page states the six-intent proxy label. It states how a
private partner project routes questions in calibrate vs evolve. It also states
which metric claims typevet may make on Banking77.

Parent: [#51](https://github.com/Alberto-Codes/typevet/issues/51),
[#53](https://github.com/Alberto-Codes/typevet/issues/53). Loader work is
[#58](https://github.com/Alberto-Codes/typevet/issues/58). Partner policy:
[eval-partner-data-policy.md](eval-partner-data-policy.md).

The partner project is the source corpus for labels and questions. Its own
documentation and its Banking77 loader define the fraud label and
`FRAUD_INTENTS`. That repository is private, so this page restates the rules
it needs.

## What Banking77 actually labels

PolyAI Banking77 (CC BY 4.0) assigns each short banking query one of 77
**intent** names. It does **not** label fraud. typevet and the partner project both use a
**proxy** binary: `fraud` or `not_fraud`.

## The six-intent proxy (partner choice)

The partner project maps six intent names to `fraud`. Every other intent is `not_fraud`.
The dataset card does not group intents. The partner project made this choice. A different
group gives different agreement numbers.

| Intent (Banking77 `category`) | Proxy label |
|---|---|
| `card_payment_not_recognised` | `fraud` |
| `cash_withdrawal_not_recognised` | `fraud` |
| `direct_debit_payment_not_recognised` | `fraud` |
| `transaction_charged_twice` | `fraud` |
| `compromised_card` | `fraud` |
| `extra_charge_on_statement` | `fraud` |
| any other intent | `not_fraud` |

Two fraud intents do not literally say “unauthorized transaction”.
`compromised_card` and `extra_charge_on_statement` still count as `fraud` under
this rule. The partner project documents that mismatch in its own
documentation.

typevet v1 adopts the **same six-intent collapse** as the partner project unless a
judgment issue overrides it ([#53](https://github.com/Alberto-Codes/typevet/issues/53#issuecomment-5841924370)
acceptance).

## Jev questions the partner project uses on banking text

The partner project defines three fraud-triage questions in `FRAUD_QUESTIONS`
plus `is_scam` from `SCAM_QUESTIONS` for the evolve agent only, in the
partner's question set:

| Question name | Type | Role on Banking77 |
|---|---|---|
| `reports_unauthorized` | Noul | “Did the customer report a transaction they did not authorize?” |
| `fraud_type` | Choice (six fraud kinds + `not_fraud` + `unclear`) | Finer fraud kind, not the 77 intents |
| `is_scam` | Noul | Scam/phishing wording; weak fit on bank-service queries |

The proxy gold label is still **six-intent `fraud` / `not_fraud`**. None of
the questions relabel the 77 intents directly.

## Calibrate vs evolve routing (partner project)

These commands measure different things. Do not merge their tables without
stating the command and the question.

### The partner `calibrate --dataset banking77` command

- Asks **one** Noul: `reports_unauthorized` on each message.
- Compares Jev’s yes-probability to the proxy label (`fraud` = positive).
- Documented in the partner project's calibration how-to.

typevet does **not** copy partner calibration ECE figures into eval claims until
verification policy [#50](https://github.com/Alberto-Codes/typevet/issues/50)
accepts them. This page defines **label and question policy**, not calibration
scores.

### The partner `evolve --dataset banking77` command

- The ADK agent calls `ask_jev` **once** per message.
- The agent picks **one** of `is_scam`, `reports_unauthorized`, or
  `fraud_type` (see `AGENT_QUESTIONS` in the partner agent module).
- The seed instruction lists all three names. gepa-adk may rewrite the
  instruction to **route by topic** (for example unauthorized charges →
  `reports_unauthorized`, exchange rates → `fraud_type`). The partner project
  records an example in its own documentation.

Probability for the evolved `Decision` depends on the question:

| Question used | Positive (`fraud`) probability rule |
|---|---|
| `reports_unauthorized` or `is_scam` | Jev Noul yes-probability |
| `fraud_type` | Sum of Choice label probabilities except `not_fraud` and `unclear` (`fraud_probability` in the partner agent module) |

Held-out **agreement** uses the same 0.5 threshold on that probability against
the proxy label. **Question mix** affects the score. Report `question_used`
distribution when you compare two evolve results.

## Choice vs Noul — metric claims typevet may make

Accepted direction from #53: **Noul-primary** on Banking77, with optional
`fraud_type` Choice as a secondary probe—not a second gold standard.

| Claim | Allowed when |
|---|---|
| Agreement (or log-loss) of **`reports_unauthorized` Noul** vs six-intent proxy | Primary v1 Banking77 metric. State proxy label explicitly. |
| Agreement of **`fraud_type` Choice** vs proxy or vs collapsed fraud probability | Secondary. State that Choice labels are **not** the 77 intents and gold is still the proxy. |
| “Calibrated on Banking77” / ECE as typevet eval headline | **Not yet.** Partner live ECE is research context only until #50. |
| Direct numeric comparison of calibrate ECE (Noul-only) to evolve agreement (routed Noul+Choice) | **No.** Different commands, questions, and aggregation rules. |
| Using `is_scam` as the default Banking77 metric | **No** for v1. Bank queries are not scam SMS. The partner project keeps `is_scam` in evolve for template parity with DIFrauD. |

When typevet exposes System One judgments ([#11](https://github.com/Alberto-Codes/typevet/issues/11)),
compile Decisions that mirror the partner names: Noul for `reports_unauthorized`,
Choice for `fraud_type`. JevBench ([#24](https://github.com/Alberto-Codes/typevet/issues/24))
remains the protocol peer. Banking77 regression tracks partner domain choices,
not JevBench substitution.

## typevet loader ([#58](https://github.com/Alberto-Codes/typevet/issues/58))

The loader is `typevet_evals.datasets.banking77`. It has the
six-intent `FRAUD_INTENTS` collapse from the partner project, test-split CSV parsing, optional
`balanced_sample` / `load_test_split(..., balanced=True)`, and
`REPORTS_UNAUTHORIZED_NOUL_SCHEMA` for the primary v1 Noul fixture. CI uses
checked-in CSV under `tests/fixtures/banking77/` (no Hugging Face Hub). Optional
`fraud_type` Choice mapping stays documented here and in the partner project; not required in
the first loader revision.

## Sampling note

The partner Banking77 loader balances `fraud` and `not_fraud` rows up to
`--limit`.
Report `limit`, `seed`, and balance when you publish agreement numbers. A
balanced sample is not the raw intent distribution.

## Related typevet pages

- [CLINC150 domain shard map](eval-clinc-shard-map.md) — complementary Choice
  shards; **no** label mapping to Banking77 or **`FRAUD_INTENTS`** ([#66](https://github.com/Alberto-Codes/typevet/issues/66)).
- [Eval partner data policy](eval-partner-data-policy.md) — public Banking77
  vs partner NBA exclusion.
- [TypeLLM, Jev and judgevet](../explanation/typellm-and-judgevet.md) — Noul,
  Choice, and probability shape for a future judgment port.
- [Glossary](glossary.md) — **Banking77 proxy label**.
