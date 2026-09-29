# go_emotions single-label conversion and prune policy

Kind: reference. This page states how typevet turns Hugging Face **go_emotions**
(multi-label) into a **Choice** eval with at most 24 labels, and which rows
survive conversion. Loader code is not on this page.

Parent: [#51](https://github.com/Alberto-Codes/typevet/issues/51),
[#67](https://github.com/Alberto-Codes/typevet/issues/67) (accepted research).
Manifest rank 6:
[Complementary eval manifest](eval-complementary-manifest.md).
Partner policy: [eval-partner-data-policy.md](eval-partner-data-policy.md).

Upstream: [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions)
(Apache 2.0), **`simplified`** config. Each row has `text` and a **list** of
emotion names. typevet complementary eval uses **Choice**, not multi-label
Score ([#67](https://github.com/Alberto-Codes/typevet/issues/67#issuecomment-5842010887)).

## State schema (v1)

| Field | Role |
|---|---|
| `text` | User message string (the Reddit comment text). |

No pair fields. No Score primitive on this set for v1.

## Upstream label space

The simplified config defines **28** class names in fixed Hub order (index order
in `datasets` `ClassLabel`). typevet **`MAX_ENUM_CHOICES`** is **24**
(`typevet.domain.decisions`). v1 **prunes four** rare labels from the enum
(frequency drop from #67), not a merge into another name.

### Pruned labels (excluded from Choice enum)

Drop these names from the compiled Choice schema and from gold after
conversion:

| Label | v1 action |
|---|---|
| `grief` | Not in enum; drop rows whose gold is only this label (after conversion). |
| `nervousness` | Same |
| `pride` | Same |
| `relief` | Same |

If a row’s surviving gold label is pruned, **exclude** the row from the eval
subset. Do not remap pruned gold into a neighbor label without a new judgment
issue.

### v1 Choice enum (24 labels, Hub order minus prune)

`admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`,
`confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`,
`embarrassment`, `excitement`, `fear`, `gratitude`, `joy`, `love`, `optimism`,
`realization`, `remorse`, `sadness`, `surprise`, `neutral`.

**`neutral` stays in the 24-enum for v1** ([#67](https://github.com/Alberto-Codes/typevet/issues/67#issuecomment-5842010887)).
Eval uses **Choice over emotion names**, not a separate Noul “is neutral”
head. Whether `neutral` should become Noul-only in a later revision remains an
open question in #67 research.

## Conversion modes

### Strict exactly-one (v1 default)

Accepted default ([#67](https://github.com/Alberto-Codes/typevet/issues/67#issuecomment-5842010887)):
**strict exactly-one gold** before prune filtering.

1. Let `L` be the upstream label list for the row.
2. **Neutral co-label rule:** If `L` contains `neutral` and at least one other
   name, remove `neutral` from `L` (do not treat neutral as co-gold with a
   specific emotion).
3. If `|L| == 1`, gold is that name. Keep the row (subject to prune filter
   below).
4. If `|L| == 0` after step 2, or `|L| > 1`, **drop** the row. Do not pick a
   winner inside strict v1.

Roughly **~83.6%** of train rows are already single-label upstream (#67
streaming stats). Strict mode keeps most of those and drops ambiguous multi-label
rows.

### Hierarchical pick (not v1 default)

#67 research contrasted **strict** vs **hierarchical pick** (choose one label
from a multi-label set by a fixed rule). typevet **does not** use hierarchical
pick as the default conversion. A future loader may expose an optional
**conversion flag**; document the flag name in the loader issue when it lands.
Until then, complementary manifest and metrics claims assume **strict
exactly-one** only.

## End-to-end row filter (v1)

Apply in order:

1. Run **strict exactly-one** conversion → gold name or drop.
2. If gold is one of the **four pruned** labels → drop.
3. If gold is not in the **24-enum** → drop (should not happen if steps 1–2
   hold).
4. Emit fixture row: `state={"text": ...}`, Choice gold ∈ 24-enum.

Report **keep rate** (rows kept / rows seen) and **split** when you publish
numbers. Subsample caps stay **TBD** in
[`evals/complementary-manifest.yaml`](https://github.com/Alberto-Codes/typevet/blob/main/evals/complementary-manifest.yaml).

## Metric claims typevet may make (v1)

| Claim | Allowed when |
|---|---|
| Choice agreement vs **strict** gold on the **24-enum** | Yes. State conversion and prune explicitly. |
| Multi-label F1 / micro-F1 on raw Hub labels | **No** for v1 complementary Choice path. |
| Score or ordinal emotion intensity | **No** ([#67](https://github.com/Alberto-Codes/typevet/issues/67#issuecomment-5842010887)). |
| Results without stating strict vs hierarchical | **No.** |

Primary primitive gap this set fills: **Choice** stress at enum size **24**
([#54](https://github.com/Alberto-Codes/typevet/issues/54) rank 6). It does
not replace JevBench ([#24](https://github.com/Alberto-Codes/typevet/issues/24)).

## Loader status

Implemented in `typevet.eval_go_emotions` ([#79](https://github.com/Alberto-Codes/typevet/issues/79)).
See [go_emotions loader and emotion Choice fixture](eval-go-emotions-loader.md).
The complementary manifest YAML may still say `not_implemented` until a manifest
update issue lands; loader code and tests are the runtime source of truth.

## Related typevet pages

- [Complementary eval manifest](eval-complementary-manifest.md) — rank 6 entry
  and YAML source of truth.
- [TypeLLM, Jev and judgevet](../explanation/typellm-and-judgevet.md) — Choice
  compilation and `MAX_ENUM_CHOICES`.
- [Glossary](glossary.md) — **go_emotions v1 Choice policy**.
