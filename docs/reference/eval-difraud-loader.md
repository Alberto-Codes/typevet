# DIFrauD loader and is_scam Noul fixture

Kind: reference. Public MIT corpus for natural binary scam detection on short
text. Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51);
loader issue [#59](https://github.com/Alberto-Codes/typevet/issues/59).

## Role in typevet

| Piece | Module / path |
|---|---|
| Test-split loader | `typevet.eval_difraud` |
| Primary Noul | `is_scam` (boolean + `return_probabilities`) |
| Versioned JSON Schema | `evals/fixtures/difraud_is_scam_noul_schema_v1.json` |
| CI JSONL subset | `tests/fixtures/difraud/sms_test_subset.jsonl` |

DIFrauD is the **natural** binary Noul fit among finvet public sets ([#53](https://github.com/Alberto-Codes/typevet/issues/53)).
Banking77 stays proxy-binary on `reports_unauthorized`; do not ask bank-fraud
questions on DIFrauD rows.

## Domain flag (v1 default: SMS)

The Hugging Face card ships separate test JSONL files per config:

| `domain` argument | Hub path segment | v1 default |
|---|---|---|
| `sms` | `sms/test.jsonl` | **yes** |
| `phishing` | `phishing/test.jsonl` | optional |
| `job_scams` | `job_scams/test.jsonl` | optional |

`typevet.eval_difraud.load_test_split` defaults to `domain="sms"`. Pass
`phishing` or `job_scams` for cross-domain regression; keep SMS as the primary
documented path until a judgment issue promotes another domain.

## Labels and class imbalance

Each JSONL row has `text` (string) and `label` (integer). The card defines
`1` as deceptive. typevet maps:

| Hub `label` | typevet `label` |
|---|---|
| `1` | `scam` |
| `0` | `legit` |

Unlike Banking77, the loader **does not balance** classes. Returned rows keep
the domain's natural scam vs legit base rate after an optional shuffle and
`limit`. Report prevalence when quoting accuracy or agreement; do not assume
50/50 unless you subsample explicitly.

## Question wording

The `is_scam` Noul instructions match finvet `SCAM_QUESTIONS["is_scam"]`: the
user message **is** the suspect text, not a customer describing fraud elsewhere.

## License and policy

DIFrauD is MIT. It is listed as a public eval dataset in
[Eval partner data policy](eval-partner-data-policy.md). collections NBA and
other partner trees stay forbidden.

## Related pages

- [Banking77 proxy and metrics](banking77-proxy-and-metrics.md) — proxy Noul,
  not DIFrauD.
- [Eval partner data policy](eval-partner-data-policy.md) — public vs partner
  data.
