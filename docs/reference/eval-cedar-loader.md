# CEDAR signature loader and signature-match request

Kind: reference. CEDAR offline signature pairs for a two-image
signature-match judgment. Parent epic:
[#304](https://github.com/Alberto-Codes/typevet/issues/304); loader issue
[#318](https://github.com/Alberto-Codes/typevet/issues/318).

## Role in typevet

| Piece | Module / path |
|---|---|
| Pair ids, slice, download and archive reader | `typevet_evals.datasets.cedar` |
| Two-image request builder | `typevet_evals.signature_match` |
| Member-name fixture (names only) | `evals/fixtures/cedar/members_excerpt.txt` |
| Default slice pair ids | `evals/fixtures/cedar/default_slice_ids.txt` |
| Unit tests | `evals/tests/unit/test_cedar_pairs.py` |
| Contract test | `evals/tests/contract/test_signature_match_contract.py` |
| Metrics, run and receipt | `typevet_evals.signature_match` (`metrics`, `runner`) |
| Metric and runner unit tests | `evals/tests/unit/test_signature_match_metrics.py` |
| Live run | `evals/tests/live/test_signature_match_live.py` |

## Source file

| File | Source URL | SHA-256 | Size |
|---|---|---|---|
| `signatures.rar` | <https://cedar.buffalo.edu/NIJ/data/signatures.rar> | `f74b859352783b82399c1be48078b79ad637160ba11f16baf92911dd5568f4d6` | 253,587,033 bytes |

CEDAR links the archive from
<https://cedar.buffalo.edu/NIJ/publications.html> under "Published Data
Sets". The download needs no sign-in.

`fetch_cedar_archive` downloads the archive into the cache when it is
absent. The loader refuses a cached or downloaded file whose SHA-256 differs
from the pinned value. A refused download leaves no file behind. A refused
cached file stays in place; remove it to fetch it again.

## Cache directory

| Order | Source |
|---|---|
| 1 | `cache_dir` argument |
| 2 | `TYPEVET_CEDAR_CACHE` environment variable |
| 3 | `~/.cache/typevet/cedar` |

The cache is outside the repository. Tests inject an HTTP client and a
temporary directory, so they do not use the network or the real file.

## Archive layout

The layout comes from the listing of the pinned archive.

| Member path | Content |
|---|---|
| `signatures/full_org/original_<writer>_<sample>.png` | Genuine signature |
| `signatures/full_forg/forgeries_<writer>_<sample>.png` | Skilled forgery |
| `signatures/Readme.txt`, `Thumbs.db` files | Not signatures; skipped |

The archive holds 55 writers, numbered 1 to 55. Each writer has 24 genuine
signatures and 24 skilled forgeries, numbered 1 to 24. Numbers have no zero
padding. `parse_member_path` holds these naming rules.

## Pairs and slice

Image 1 of every pair is a genuine signature.

| Pair kind | Image 2 |
|---|---|
| `genuine_genuine` | Another genuine signature by the same writer |
| `genuine_skilled` | A skilled forgery of the same writer's signature |
| `genuine_random` | A genuine signature by another writer |

`select_balanced_slice` returns 60 pairs of each kind by default, with seed
0. Each kind visits the 55 writers in a seeded order, one pair per writer
per round. The default slice uses every writer for each kind, and 5 writers
twice. SHA-256 keys set every order, so the same seed gives the same slice
on every Python version. `default_slice_ids.txt` pins the 180 pair ids of the
default slice.

`read_members` reads image bytes by member path. It runs the `unrar` tool
once to extract the requested members into a temporary directory, then
removes that directory. It raises `CedarToolError` when `unrar` is not on
`PATH`. It refuses a name that is not a signature member path with
`ValueError`, before it runs the tool. It reports a member that is not in
the archive with `KeyError`, also when `unrar` exits with status 10. It
raises `FileNotFoundError` when the archive file does not exist. Tests
inject a fake command, so they need no RAR tool.

## Judgment request

`build_signature_match_request` makes one typevet judgment per pair. Image
1 is the genuine reference and image 2 is the questioned signature. The
state and the questions do not name the writer, the file or the pair kind.

| Question id | Type | Answer |
|---|---|---|
| `same_writer` | `Noul` | Probability that the same person wrote both signatures |
| `verdict` | `Choice` | `same_writer`, `different_writer`, `skilled_forgery_suspected` or `cannot_tell` |
| `image_quality` | `Score` | 0 to 4, how clearly image 2 shows the signature |

`judge_signature_match` sends the request to a `JudgmentPort` with both
images in order. The contract test proves the wiring with a fake scorer. It
says nothing about model quality. The `same_writer` value is model
confidence. It is not a match percentage or a forensic score.

## Live run and receipt

Issue [#319](https://github.com/Alberto-Codes/typevet/issues/319) runs the
default slice once per backend. `run_signature_match` sends one judgment
per pair and stops at the first backend failure. The receipt records that
failure.

A verdict of `different_writer` or `skilled_forgery_suspected` says
"different writer". `cannot_tell` says neither side.

| Metric | Definition |
|---|---|
| `accuracy` | Share of verdicts on the right same-writer side. `cannot_tell` is always wrong. |
| `kind_accuracy` | Share of verdicts that name the pair kind: `same_writer`, `skilled_forgery_suspected` or `different_writer` |
| `roc_auc` | ROC-AUC of `same_writer`, Mann-Whitney with average ranks for ties |
| `roc_auc_by_negative_kind` | ROC-AUC of genuine pairs against skilled pairs, and against random pairs |
| `ece` | Expected calibration error over ten equal-width bins |
| `reliability` | The ten bins: count, mean confidence, share of same-writer pairs |
| `cannot_tell_rate` | Share of `cannot_tell` verdicts |
| `skilled_false_accept` | On skilled pairs only: share with `same_writer` at or above 0.5 (`noul_rate`), and share with the `same_writer` verdict (`verdict_rate`) |
| `noul_choice_agreement` | Share of pairs where the `same_writer` side (at or above 0.5) and the verdict side agree. `cannot_tell` pairs are counted apart and are not in the rate. |
| `by_kind` | For each pair kind: pair count, accuracy, accept rates, verdict counts and `image_quality` level counts |

The `same_writer` value is model confidence. It is not a match percentage
or a forensic score.

| Variable | Use |
|---|---|
| `TYPEVET_SIGNATURE_MATCH_RECEIPT` | Receipt path. It must name a new file. The test skips when it is not set. |
| `TYPEVET_CEDAR_CACHE` | Cache directory for the archive |
| `TYPEVET_BACKEND` | `llama_cpp` (default) or `vllm`. Any other value, `fake` included, fails the test before any network call. See [backend selection](configuration.md#backend-selection). |
| `TYPEVET_LLAMA__MULTIMODAL_MODEL` | llama.cpp model; the test default is `gemma-4-31b-kv9-q4km-mm` |
| `TYPEVET_VLLM__BASE_URL`, `TYPEVET_VLLM__MODEL`, `TYPEVET_VLLM__API_KEY`, `TYPEVET_VLLM__USER_AGENT` | vLLM session |
| `TYPEVET_VLLM_MODEL_REVISION` | Served weights revision. Required when `TYPEVET_BACKEND` is `vllm`. The test fails before any network call when it is not set. |
| `TYPEVET_SIGNATURE_MATCH_PER_KIND` | Smaller slice for a smoke run |
| `TYPEVET_GIT_STATUS_PORCELAIN` | Porcelain status text for the working-tree fingerprint |

```bash
TYPEVET_GIT_STATUS_PORCELAIN="$(git status --porcelain)" \
  TYPEVET_SIGNATURE_MATCH_RECEIPT=evals/fixtures/cedar/receipts/signature_match_llama_cpp.json \
  uv run pytest evals/tests/live/test_signature_match_live.py -m live -q -s
```

A full run first checks the slice pair ids against `default_slice_ids.txt`.
The receipt holds pair ids, pair kinds, typed answers, latency per pair, the
metrics and the pins. The pins are the archive SHA-256, the slice seed and
the pairs per kind. They also include the SHA-256 of the slice pair ids and
the server facts. The experiment identity is a separate `identity` key.
A llama.cpp receipt pins the server build, the model alias and `n_ctx`, but no GGUF hash, as the face-match receipt does. A vLLM
receipt also pins the served weights revision as `model_revision`. The
receipt holds no image bytes. The test refuses to write a receipt that holds
the vLLM key or an auth header. Receipts are in
`evals/fixtures/cedar/receipts/`.

### Throughput block

The receipt also holds the `throughput` key from
[#335](https://github.com/Alberto-Codes/typevet/issues/335). It records the concurrency, the wall time
and the judgments and images per second. It also records the latency
percentiles, the `discarded` count and, on vLLM, the `/metrics` changes
over the run. The
image count is 2 per signature pair. The `stopped` record also holds `discarded`. The
[LFW reference](eval-lfw-loader.md#throughput-block) lists each key. The
[receipt blocks page](eval-receipt-blocks.md) describes the `server` and
`server_args` blocks. The
code fingerprint in the experiment identity includes `face_match/pool.py`
and `serving_metrics.py`.

## Licence and policy

The CEDAR page states no licence. typevet uses the data for research only
and fetches it from CEDAR at run time.

The repository stores no signature bytes. Fixtures hold member names and
pair ids only, and tests use synthetic solid-colour PNG images. Do not
commit images from the archive, and do not write them into receipts.

## Related pages

- [Eval partner data policy](eval-partner-data-policy.md): public and
  partner data.
- [LFW loader](eval-lfw-loader.md): the face-match loader that this loader
  follows.
- [Receipt blocks](eval-receipt-blocks.md): the serving-metrics and
  `server_args` blocks.
- [Two-image signature comparison](../explanation/two-image-signature-comparison.md):
  what the signature-match runs measured and their limits.
