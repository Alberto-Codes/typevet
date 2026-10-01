# Synthetic checks and the check-match run

Kind: reference. Generated paper checks, judged against a synthetic check
register row. Parent epic:
[#303](https://github.com/Alberto-Codes/typevet/issues/303); generator issue
[#315](https://github.com/Alberto-Codes/typevet/issues/315); run issue
[#316](https://github.com/Alberto-Codes/typevet/issues/316).

## Role in typevet

| Piece | Module / path |
|---|---|
| Register rows, variants and expected labels | `typevet_evals.check_match.cases` |
| Renders and contact sheet | `typevet_evals.check_match.render` |
| Offline contact-sheet command | `typevet_evals.cli.check_sheets` |
| One-image request builder | `typevet_evals.check_match.request` |
| Metric rules | `typevet_evals.check_match.metrics` |
| Run and receipt | `typevet_evals.check_match.runner` |
| Metric and runner unit tests | `evals/tests/unit/test_check_match_metrics.py` |
| Contract test | `evals/tests/contract/test_check_match_contract.py` |
| Live run | `evals/tests/live/test_check_match_live.py` |
| Option-order study (#105) | `typevet_evals.check_match.orderings` |
| Option-order unit tests | `evals/tests/unit/test_check_match_orderings.py` |
| Option-order live run | `evals/tests/live/test_check_match_orderings_live.py` |

## Generator

`check_cases` returns 20 register rows times 7 variants, so 140 cases. The
default seed is 0. SHA-256 keys set every draw, so one seed gives one slice.

Each register row has a check number, a date, a payee and an amount. Payees
come from a fixed list of invented names. The case id is
`r<row>:<variant>`, for example `r00:clean`.

Every check is not negotiable by construction:

- The routing number fails the ABA check digit.
- The account number prints as zeros.
- A `SPECIMEN` mark crosses the face.
- A `VOID` mark crosses the signature line on every render.
- The bank name says that the bank is not real.
- Signatures are seeded synthetic strokes, never a real signature.

Renders use the bundled Pillow font only. The repository stores no render.
Renders are byte-identical for one case within one Pillow build.

## Contact sheets without a model call

This command writes the contact sheets for one seed. It makes no model
call. Put the output directory outside the repository or under
`scratchpad/`, which the repository ignores.

```bash
uv run python -m typevet_evals.cli.check_sheets --seed 1 --out scratchpad/check_sheets_seed1
```

| Option | Use |
|---|---|
| `--seed` | Generator seed. The default is 0, the #316 slice. ASCII digits only, with outer spaces allowed; a blank value gives 0, as `TYPEVET_CHECK_MATCH_SEED` does. |
| `--rows` | Register rows, 1 to 20. The default is 20. |
| `--out` | Output directory. The command makes it when it is missing. |

The command writes one PNG per register row, `seed<seed>_row<row>.png`, for
example `seed1_row00.png`. Each sheet is one `write_contact_sheet` grid of
the 7 variants of that row, with the expected labels under each render. The
command prints one path per line. It exits 0 when it writes every sheet and
2 on a usage error.

## Variants and expected labels

These are amendment A1 of the #303 design. The owner signed off the labels
on a contact sheet of all 140 renders (amendment A2).

| Variant | Accepted `verdict` | Payee truth | Amounts truth |
|---|---|---|---|
| `clean` | `consistent` | yes | yes |
| `payee_changed` | `payee_mismatch` | no | yes |
| `written_amount_changed` | `amount_mismatch` | yes | no |
| `both_amounts_changed` | `amount_mismatch` | yes | no |
| `wrong_date` | `date_mismatch` | yes | yes |
| `unsigned` | `unsigned` | yes | yes |
| `low_legibility` | `consistent` or `cannot_tell` | yes | yes |

In `written_amount_changed`, the number amount equals the register. The
`low_legibility` variant has clean content and a Gaussian blur.

## Judgment request

`build_check_match_request` makes one typevet judgment per case. The state
holds the register row as text. The media holds one check image.

| Question id | Type | Answer |
|---|---|---|
| `payee_matches` | `Noul` | Probability that the check names the register payee |
| `amounts_match` | `Noul` | Probability that both check amounts equal the register amount |
| `verdict` | `Choice` | `consistent`, `payee_mismatch`, `amount_mismatch`, `date_mismatch`, `unsigned` or `cannot_tell` |
| `legibility` | `Score` | 0 to 4, how clearly the check text can be read |

`judge_check_match` sends the request to a `JudgmentPort`.

## Run

`run_check_match` sends one judgment per case, in slice order. A backend
failure (`GenerationError`) stops the run. The run keeps the earlier
outcomes and records the index, case id and error class of the failure.

## Metrics

`check_match_metrics` computes these values. A rate is `null` when no case
counts.

| Metric | Definition |
|---|---|
| `accuracy` | Share of cases whose `verdict` is in the accepted set |
| `accuracy_by_class` | Accuracy per accepted set, for example `cannot_tell\|consistent` |
| `accuracy_by_variant` | Accuracy per variant; `unsigned` shows separately |
| `false_clear_rate` | Share of counted cases answered `consistent` |
| `false_clear_by_variant` | The same rate for each counted variant |
| `cannot_tell_rate`, `cannot_tell_by_variant` | Share of `cannot_tell` verdicts, overall and per variant |
| `nouls.<id>.roc_auc` | ROC-AUC of the `Noul` against its truth, average ranks for ties |
| `nouls.<id>.ece` | Expected calibration error over ten equal-width bins |
| `nouls.<id>.reliability` | The ten bins: count, mean confidence, share of true cases |
| `score_by_variant` | Mean `legibility` value and the level counts per variant |
| `legibility_gap_clean_minus_low` | Clean mean `legibility` minus the low-legibility mean |
| `noul_choice_agreement` | Share of counted cases whose `Noul` answers agree with the `verdict` |

The false-clear rate counts the five variants whose accepted set does not
hold `consistent`. These are `payee_changed`, both amount variants,
`wrong_date` and `unsigned`. The `low_legibility` variant never counts.

The agreement rule uses a limit of 0.5. A `Noul` below 0.5 says "mismatch".

| `verdict` | Agrees when |
|---|---|
| `payee_mismatch` | The payee `Noul` says mismatch |
| `amount_mismatch` | The amounts `Noul` says mismatch |
| `consistent`, `date_mismatch`, `unsigned` | Neither `Noul` says mismatch |
| `cannot_tell` | Not counted |

`Noul` values are model confidence. They are not calibrated match
percentages. Generated checks are not evidence about real checks, fraud or
counterfeits.

## Live run and receipt

| Variable | Use |
|---|---|
| `TYPEVET_CHECK_MATCH_RECEIPT` | Receipt path. It must name a new file. The test skips when it is not set. |
| `TYPEVET_CHECK_MATCH_SEED` | Generator seed, a non-negative integer. The default is 0, the #316 slice. The receipt records it as `generator_seed`. |
| `TYPEVET_CHECK_MATCH_ROWS` | Register rows, 1 to 20, for a smoke run. Each row gives all 7 variants. |
| `TYPEVET_REQUIRE_LIVE` | When true, a missing receipt path fails the test |
| `TYPEVET_BACKEND` | `llama_cpp` (default) or `vllm` |
| `TYPEVET_LLAMA__MULTIMODAL_MODEL` | llama.cpp model; the test default is `gemma-4-31b-kv9-q4km-mm` |
| `TYPEVET_VLLM__BASE_URL`, `TYPEVET_VLLM__MODEL`, `TYPEVET_VLLM__API_KEY`, `TYPEVET_VLLM__USER_AGENT` | vLLM session |
| `TYPEVET_VLLM_MODEL_REVISION` | Served weights revision. Required when `TYPEVET_BACKEND` is `vllm`. |
| `TYPEVET_GIT_STATUS_PORCELAIN` | Porcelain status text for the working-tree fingerprint |

The test checks the receipt path, the row count, the seed and the vLLM
revision before any network call.

```bash
TYPEVET_GIT_STATUS_PORCELAIN="$(git status --porcelain)" \
  TYPEVET_CHECK_MATCH_RECEIPT=evals/fixtures/checks/receipts/check_match_llama_cpp_receipt.json \
  uv run pytest evals/tests/live/test_check_match_live.py -m live -q -s
```

Each receipt case holds the case id, the variant, the expected labels, the
typed answers, and the agreement flag. It also holds the render SHA-256,
the latency and the prompt tokens. The receipt also holds the metrics and
the stopping failure.

| Pin | Meaning |
|---|---|
| `generator_seed`, `register_rows`, `variants_per_row` | The slice |
| `pillow_version` | Pillow build that drew the renders |
| `slice_sha256` | SHA-256 over each case id and render digest |
| `server` | Served model entry and server build |
| `model_revision` | Served weights revision, vLLM only |

The receipt also holds the experiment identity with the git fingerprint.
The receipt holds no image bytes. The test refuses a receipt that holds the
vLLM key or an auth header.

### Throughput block

The receipt also holds the `throughput` key from
[#335](https://github.com/Alberto-Codes/typevet/issues/335). It records the concurrency, the wall time
and the judgments and images per second. It also records the latency
percentiles, the `discarded` count and, on vLLM, the `/metrics` changes
over the run. The
image count is 1 per check. The `stopped` record also holds `discarded`. The
[LFW reference](eval-lfw-loader.md#throughput-block) lists each key. The
[receipt blocks page](eval-receipt-blocks.md) describes the `server` and
`server_args` blocks. The
code fingerprint in the experiment identity includes `face_match/pool.py`
and `serving_metrics.py`.

### Option-order run

The option-order live test scores the seed-1 verdict once per balanced
ordering. Issue [#105](https://github.com/Alberto-Codes/typevet/issues/105)
holds the rule.

| Variable | Use |
|---|---|
| `TYPEVET_CHECK_ORDERINGS_RECEIPT` | Receipt path. It must name a new file. The test skips when it is not set. |
| `TYPEVET_CHECK_ORDERINGS_ROWS` | Register rows, 1 to 20. The default is 3, the 21-case receipt. ASCII digits only, with outer spaces allowed. |

The test sends rows × 7 × 6 requests. Twenty rows give 140 cases and 840
requests. A refused rows value gives an error that names the variable, not
the value.

| Statistics key | Meaning |
|---|---|
| `spread_interval` | 95% bootstrap interval of the position spread, as two floats |
| `accuracy_difference_interval` | 95% bootstrap interval of averaged minus single-order accuracy, as two floats |
| `bootstrap` | `resamples` (1000) and `seed` (0) of the resampling over cases |

The decision label uses the point estimates only. The intervals are
reported, not part of the rule.

### Repeat runs

Issue [#357](https://github.com/Alberto-Codes/typevet/issues/357) holds the
pre-registered rule. The repeats use the main live test with a new receipt
path each time.

| Receipt | Run |
|---|---|
| `check_match_llama_cpp_receipt.json` | Run 0, with other code-path digests |
| `check_match_llama_cpp_repeat1.json` to `check_match_llama_cpp_repeat5.json` | Repeats 1 to 5 of seed 0 on llama.cpp |

`typevet_evals.check_match.repeats` reads the receipts. `repeat_summary(paths)`
returns these keys.

| Summary key | Meaning |
|---|---|
| `runs` | Per run: name, baseline commit, accuracy, false-clear rate, ECE per `Noul`, wall seconds |
| `runs[].included` | False when the code-path digests differ from the majority |
| `runs[].agreement`, `runs[].max_abs_difference` | Verdict agreement and largest probability change against the first included run |
| `excluded` | Run name and the names of the differing digests, never the values |
| `statistics` | Mean, sample `sd`, `min`, `max` and `range` of accuracy, false-clear rate and each ECE |
| `accuracy_interval` | 95% percentile interval over pooled case rows: 10,000 resamples, seed 0 |
| `agreement` | Share of cases with one verdict in every included run |
| `max_abs_difference` | Largest change of any recorded probability over included runs |
| `stable` | True when the accuracy range and every ECE range are below 0.01 |

## llama.cpp media marker

The llama.cpp router gives each model load a new random media marker. The
session reads the marker once, when it opens. A model swap by another
router client can make that marker stale. The server then refuses each
request with HTTP 400 "Failed to tokenize prompt". Issue
[#322](https://github.com/Alberto-Codes/typevet/issues/322) tracks the fix.
Until then, do not share the router with other clients during a run.

## Related pages

- [Eval partner data policy](eval-partner-data-policy.md): public, partner
  and generated data.
- [LFW loader](eval-lfw-loader.md): the face-match run that this run
  follows.
- [CEDAR loader](eval-cedar-loader.md): the signature-match request.
- [Receipt blocks](eval-receipt-blocks.md): the serving-metrics and
  `server_args` blocks.
- [Check images against a synthetic register](../explanation/check-register-matching.md):
  what the check-match runs measured and their limits.
