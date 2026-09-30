# LFW View 2 loader and face-match request

Kind: reference. Labeled Faces in the Wild (LFW) View 2 face pairs for a
two-image face-match judgment. Parent epic:
[#292](https://github.com/Alberto-Codes/typevet/issues/292); loader issue
[#300](https://github.com/Alberto-Codes/typevet/issues/300).

## Role in typevet

| Piece | Module / path |
|---|---|
| Pairs loader, slice and archive reader | `typevet_evals.datasets.lfw` |
| Two-image request builder | `typevet_evals.face_match` |
| Pairs fixture (synthetic names and numbers only) | `evals/fixtures/lfw/pairs_excerpt.txt` |
| Unit tests | `evals/tests/unit/test_lfw_pairs.py` |
| Contract test | `evals/tests/contract/test_face_match_contract.py` |
| Metrics, run and receipt | `typevet_evals.face_match` (`metrics`, `runner`) |
| Metric and runner unit tests | `evals/tests/unit/test_face_match_metrics.py` |
| Live run | `evals/tests/live/test_face_match_live.py` |

## Source files

| File | Source URL | SHA-256 | Size |
|---|---|---|---|
| `pairs.txt` | <https://ndownloader.figshare.com/files/5976006> | `ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592` | 155,335 bytes |
| `lfw-funneled.tgz` | <https://ndownloader.figshare.com/files/5976015> | `b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a` | 243,346,528 bytes |

`fetch_lfw_files` downloads each file into the cache when it is absent. The
loader refuses a cached or downloaded file whose SHA-256 differs from the
pinned value. A refused download leaves no file behind. A refused cached file
stays in place; remove it to fetch it again.

## Cache directory

| Order | Source |
|---|---|
| 1 | `cache_dir` argument |
| 2 | `TYPEVET_LFW_CACHE` environment variable |
| 3 | `~/.cache/typevet/lfw` |

The cache is outside the repository. Tests inject an HTTP client and a
temporary directory, so they do not use the network or the real files.

## Pairs and slice

`pairs.txt` starts with the header `10` and `300`, separated by a tab. Each
of the ten folds holds 300 same-person lines (`name`, `i`, `j`), then 300
different-person lines (`name1`, `i`, `name2`, `j`). `parse_pairs` returns
6,000 `LfwPair` values with the fold, the left face, the right face and the
gold `same_person` label.

`select_balanced_slice` returns 100 same-person pairs and 100
different-person pairs by default, with seed 0. Each class takes pairs from
the folds in turn, so the default slice holds 20 pairs from each fold.
SHA-256 keys order each fold and the result, so the same seed gives the same
slice on every Python version.

`read_members` reads image bytes from the archive by member path, for example
`lfw_funneled/<name>/<name>_0001.jpg`. It reads the archive once and extracts
nothing to disk.

## Judgment request

`build_face_match_request` makes one typevet judgment per pair. Image 1 is
the left face and image 2 is the right face. The state and the questions do
not name the people in the pair.

| Question id | Type | Answer |
|---|---|---|
| `same_person` | `Noul` | Probability that the main faces are the same person |
| `verdict` | `Choice` | `same_person`, `different_person` or `cannot_tell` |
| `face_visibility` | `Score` | 0 to 4, how clearly image 2 shows the main face |

`judge_face_match` sends the request to a `JudgmentPort` with both images in
order. The contract test proves the wiring with a fake scorer. It says
nothing about model quality.

## Live run and receipt

Issue [#301](https://github.com/Alberto-Codes/typevet/issues/301) runs the
default slice once per backend. `run_face_match` sends one judgment per
pair and stops at the first backend failure. The receipt records that
failure.

| Metric | Definition |
|---|---|
| `accuracy` | Share of right `verdict` answers. `cannot_tell` is always wrong. |
| `roc_auc` | ROC-AUC of `same_person`, Mann-Whitney with average ranks for ties |
| `ece` | Expected calibration error over ten equal-width bins |
| `reliability` | The ten bins: count, mean confidence, share of same-person pairs |
| `cannot_tell_rate` | Share of `cannot_tell` verdicts |
| `score_distribution` | Count of each `face_visibility` level, by gold label |

The `same_person` value is model confidence. It is not a calibrated match
percentage.

| Variable | Use |
|---|---|
| `TYPEVET_FACE_MATCH_RECEIPT` | Receipt path. It must name a new file. The test skips when it is not set. |
| `TYPEVET_BACKEND` | `llama_cpp` (default) or `vllm` |
| `TYPEVET_LLAMA__MULTIMODAL_MODEL` | llama.cpp model; the test default is `gemma-4-31b-kv9-q4km-mm` |
| `TYPEVET_VLLM__BASE_URL`, `TYPEVET_VLLM__MODEL`, `TYPEVET_VLLM__API_KEY`, `TYPEVET_VLLM__USER_AGENT` | vLLM session |
| `TYPEVET_FACE_MATCH_PER_CLASS` | Smaller slice for a smoke run |
| `TYPEVET_GIT_STATUS_PORCELAIN` | Porcelain status text for the working-tree fingerprint |

```bash
TYPEVET_GIT_STATUS_PORCELAIN="$(git status --porcelain)" \
  TYPEVET_FACE_MATCH_RECEIPT=evals/fixtures/lfw/receipts/face_match_llama_cpp_receipt.json \
  uv run pytest evals/tests/live/test_face_match_live.py -m live -q -s
```

The receipt holds pair ids, gold labels, typed answers, latency per pair,
the metrics and the pins. The pins are the LFW file SHA-256 values, the
slice seed, the server build and the experiment identity. The receipt holds
no image bytes. The test refuses to write
a receipt that holds the vLLM key or an auth header. Receipts are in
`evals/fixtures/lfw/receipts/`.

## Licence and policy

LFW has no formal licence. The photographers keep the image copyright. The
dataset is for research use.

The repository stores no face bytes. Fixtures hold names and image numbers
only, and tests use synthetic solid-colour PNG images. Do not commit images
from the archive, and do not write them into receipts.

## Related pages

- [Eval partner data policy](eval-partner-data-policy.md): public and
  partner data.
