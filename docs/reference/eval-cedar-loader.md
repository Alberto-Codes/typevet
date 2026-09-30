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
