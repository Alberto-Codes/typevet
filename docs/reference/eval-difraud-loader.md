# DIFrauD loader and is_scam Noul fixture

Kind: reference. Public MIT corpus for natural binary scam detection on short
text. Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51);
loader issue [#59](https://github.com/Alberto-Codes/typevet/issues/59).

## Role in typevet

| Piece | Module / path |
|---|---|
| Test-split loader | `typevet_evals.datasets.difraud` |
| Primary Noul | `is_scam` (boolean + `return_probabilities`) |
| Versioned JSON Schema | `evals/fixtures/difraud_is_scam_noul_schema_v1.json` |
| CI JSONL subset | `tests/fixtures/difraud/sms_test_subset.jsonl` |

DIFrauD is the **natural** binary Noul fit among the partner public sets ([#53](https://github.com/Alberto-Codes/typevet/issues/53)).
Banking77 stays proxy-binary on `reports_unauthorized`; do not ask bank-fraud
questions on DIFrauD rows.

## Domain flag (v1 default: SMS)

The Hugging Face card ships separate test JSONL files per config:

| `domain` argument | Hub path segment | v1 default |
|---|---|---|
| `sms` | `sms/test.jsonl` | **yes** |
| `phishing` | `phishing/test.jsonl` | optional |
| `job_scams` | `job_scams/test.jsonl` | optional |

`typevet_evals.datasets.difraud.load_test_split` defaults to `domain="sms"`. Pass
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

## SMS train, validation and held-out splits

`typevet_evals.datasets.difraud.load_splits` downloads SMS `train.jsonl`,
`validation.jsonl` and `test.jsonl` at the pinned Hub revision
`aaaf94b336c563a14806bb4f3f58727bed9ed8d4` (`PINNED_REVISION`).
`build_splits` does the same work on JSONL text that you supply.
Issue [#307](https://github.com/Alberto-Codes/typevet/issues/307) defines the rules.

| Rule | Behaviour |
|---|---|
| Record id | `record_id(text)`: `sha256:` and the first 16 hex characters of the SHA-256 of the exact text. Upstream rows have no id field. Rows with the same text have the same id. |
| No shared id | A record stays in the first split where it appears, in the order train, validation, held-out. A split keeps only the first row of each id. If a later row with the same text has a different label, that label is lost. |
| Held-out | `test.jsonl` without the 500 rows that [#236](https://github.com/Alberto-Codes/typevet/issues/236) measured. |
| #236 rows | `prior_measured_ids` calls `load_test_split(limit=500, seed=0)`. `typevet_evals.throughput.public_workload` makes the same call. |
| Order | `seed` sets the hash-based order of each split. The #236 selection always uses seed 0. |

The #236 run downloaded `test.jsonl` from `main`, not from a pinned revision.
The exclusion is correct only if `main` then had the same `test.jsonl` as `PINNED_REVISION`.
On 2026-09-30, `main` was `PINNED_REVISION` (last change 2024-08-02), and both give the same `test.jsonl` SHA-256.
The reproduced 500 rows hold 92 scam rows, as the #236 receipt states.

At `PINNED_REVISION` with seed 0, the splits have these sizes. No text occurs twice, in a split or across splits.

| Split | Rows | Scam |
|---|---|---|
| Train | 5,259 | 1,019 |
| Validation | 657 | 127 |
| Held-out | 158 | 36 |

Held-out records keep `split="test"`, the upstream file name.

## Question wording

The `is_scam` Noul instructions match the partner `SCAM_QUESTIONS["is_scam"]`: the
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
