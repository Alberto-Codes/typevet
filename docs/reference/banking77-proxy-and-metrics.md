# Banking77 proxy label and Choice vs Noul metrics

Kind: reference. This page states the six-intent proxy label, how finvet
routes questions in calibrate vs evolve, and which metric claims typevet may
make on Banking77.

Parent: [#51](https://github.com/Alberto-Codes/typevet/issues/51),
[#53](https://github.com/Alberto-Codes/typevet/issues/53). Loader work is
[#58](https://github.com/Alberto-Codes/typevet/issues/58). Partner policy:
[eval-partner-data-policy.md](eval-partner-data-policy.md).

finvet is the source corpus for labels and questions. See finvet
[The Banking77 fraud label](https://github.com/Alberto-Codes/finvet/blob/main/docs/reference/datasets.md#the-banking77-fraud-label),
[Why finvet evolves on Banking77](https://github.com/Alberto-Codes/finvet/blob/main/docs/explanation/why-banking77.md),
and `finvet.data.banking77` (`FRAUD_INTENTS` in
[`banking77.py`](https://github.com/Alberto-Codes/finvet/blob/main/src/finvet/data/banking77.py)).

## What Banking77 actually labels

PolyAI Banking77 (CC BY 4.0) assigns each short banking query one of 77
**intent** names. It does **not** label fraud. typevet and finvet both use a
**proxy** binary: `fraud` or `not_fraud`.

## The six-intent proxy (finvet choice)

finvet maps six intent names to `fraud`. Every other intent is `not_fraud`.
The dataset card does not group intents. finvet made this choice. A different
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
this rule. finvet documents that mismatch in
[why-banking77](https://github.com/Alberto-Codes/finvet/blob/main/docs/explanation/why-banking77.md).

typevet v1 adopts the **same six-intent collapse** as finvet unless a
judgment issue overrides it ([#53](https://github.com/Alberto-Codes/typevet/issues/53#issuecomment-5841924370)
acceptance).

## Jev questions finvet uses on banking text

finvet defines three fraud-triage questions in `FRAUD_QUESTIONS` plus
`is_scam` from `SCAM_QUESTIONS` for the evolve agent only
([`questions.py`](https://github.com/Alberto-Codes/finvet/blob/main/src/finvet/questions.py)):

| Question name | Type | Role on Banking77 |
|---|---|---|
| `reports_unauthorized` | Noul | “Did the customer report a transaction they did not authorize?” |
| `fraud_type` | Choice (six fraud kinds + `not_fraud` + `unclear`) | Finer fraud kind, not the 77 intents |
| `is_scam` | Noul | Scam/phishing wording; weak fit on bank-service queries |

The proxy gold label is still **six-intent `fraud` / `not_fraud`**. None of
the questions relabel the 77 intents directly.

## Calibrate vs evolve routing (finvet)

These commands measure different things. Do not merge their tables without
stating the command and the question.

### `finvet calibrate --dataset banking77`

- Asks **one** Noul: `reports_unauthorized` on each message.
- Compares Jev’s yes-probability to the proxy label (`fraud` = positive).
- Documented in finvet [Run a calibration](https://github.com/Alberto-Codes/finvet/blob/main/docs/how-to/run-a-calibration.md).

typevet does **not** copy finvet calibration ECE figures into eval claims until
verification policy [#50](https://github.com/Alberto-Codes/typevet/issues/50)
accepts them. This page defines **label and question policy**, not calibration
scores.

### `finvet evolve --dataset banking77`

- The ADK agent calls `ask_jev` **once** per message.
- The agent picks **one** of `is_scam`, `reports_unauthorized`, or
  `fraud_type` (see `AGENT_QUESTIONS` in
  [`agent.py`](https://github.com/Alberto-Codes/finvet/blob/main/src/finvet/agent.py)).
- The seed instruction lists all three names. gepa-adk may rewrite the
  instruction to **route by topic** (for example unauthorized charges →
  `reports_unauthorized`, exchange rates → `fraud_type`). finvet records an
  example in
  [why-banking77](https://github.com/Alberto-Codes/finvet/blob/main/docs/explanation/why-banking77.md#issue-2-run).

Probability for the evolved `Decision` depends on the question:

| Question used | Positive (`fraud`) probability rule |
|---|---|
| `reports_unauthorized` or `is_scam` | Jev Noul yes-probability |
| `fraud_type` | Sum of Choice label probabilities except `not_fraud` and `unclear` (`fraud_probability` in finvet `agent.py`) |

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
| “Calibrated on Banking77” / ECE as typevet eval headline | **Not yet.** finvet live ECE is research context only until #50. |
| Direct numeric comparison of calibrate ECE (Noul-only) to evolve agreement (routed Noul+Choice) | **No.** Different commands, questions, and aggregation rules. |
| Using `is_scam` as the default Banking77 metric | **No** for v1. Bank queries are not scam SMS. finvet keeps `is_scam` in evolve for template parity with DIFrauD. |

When typevet exposes System One judgments ([#11](https://github.com/Alberto-Codes/typevet/issues/11)),
compile Decisions that mirror finvet names: Noul for `reports_unauthorized`,
Choice for `fraud_type`. JevBench ([#24](https://github.com/Alberto-Codes/typevet/issues/24))
remains the protocol peer. Banking77 regression tracks finvet domain choices,
not JevBench substitution.

## Sampling note

finvet `banking77.load` balances `fraud` and `not_fraud` rows up to `--limit`
([`banking77.py`](https://github.com/Alberto-Codes/finvet/blob/main/src/finvet/data/banking77.py)).
Report `limit`, `seed`, and balance when you publish agreement numbers. A
balanced sample is not the raw intent distribution.

## Related typevet pages

- [Eval partner data policy](eval-partner-data-policy.md) — public Banking77
  vs partner NBA exclusion.
- [TypeLLM, Jev and judgevet](../explanation/typellm-and-judgevet.md) — Noul,
  Choice, and probability shape for a future judgment port.
- [Glossary](glossary.md) — **Banking77 proxy label**.
