# Eval partner data policy

Kind: reference. This page states which finvet datasets typevet may ship in public
artifacts and which paths CI must reject.

Parent non-goals: [#51](https://github.com/Alberto-Codes/typevet/issues/51),
[#53](https://github.com/Alberto-Codes/typevet/issues/53). Guard issue:
[#61](https://github.com/Alberto-Codes/typevet/issues/61).

## Public eval datasets (v1 direction)

typevet may adopt **public** finvet loaders and fixtures when a child issue
accepts them. Examples from the #53 research return:

| Dataset | License | typevet role |
|---|---|---|
| Banking77 | CC BY 4.0 | Proxy binary Noul / optional Choice |
| DIFrauD | MIT | Natural binary Noul (`is_scam`); see [DIFrauD loader](eval-difraud-loader.md) |

CFPB and synth collections are **seed-only**; see
[CFPB and synth seed-only](eval-cfpb-synth-seed-only.md). ABCD and
UCI SMS stay parked until a later scan adopts them.

## Other public eval datasets

These datasets do not come from finvet. Issue
[#292](https://github.com/Alberto-Codes/typevet/issues/292) adopts LFW.
Issue [#304](https://github.com/Alberto-Codes/typevet/issues/304) adopts
CEDAR.

| Dataset | License | typevet role |
|---|---|---|
| LFW View 2 | No formal licence; photographers keep image copyright; research use only | Two-image face match; no face bytes stored in the repository; see [LFW loader](eval-lfw-loader.md) |
| CEDAR signatures | No licence stated on the [source page](https://cedar.buffalo.edu/NIJ/publications.html); research use only | Two-image signature match. Run-time fetch with a pinned SHA-256 into a cache outside the repository. Fixtures hold ids only. No signature bytes stored in the repository; see [CEDAR loader](eval-cedar-loader.md) |

## Collections NBA — never in public typevet

**collections NBA** is finvet’s partner next-best-action split. finvet keeps it
**local only** under git-ignored `data/collections_nba/`. The license is
partner data; finvet does **not** redistribute it
([finvet datasets](https://github.com/Alberto-Codes/finvet/blob/main/docs/reference/datasets.md)).

typevet must **not**:

- commit, bundle, or publish partner jsonl or derived shards;
- add wheel/sdist paths that include `collections_nba` or `data/collections_nba`;
- vendor finvet’s `collections_nba` loader or partner `nba` reward package;
- document a “download this split” path for collections NBA in typevet artifacts.

Developers may still run finvet locally with their own partner copy. That
workflow stays outside this repo.

## Forbidden markers (machine check)

CI runs `typevet_evals.datasets.partner_guard` via pytest. A tracked
path or packaging line must not contain any marker below. The exceptions are
the allowlisted guard and policy files named in that module.

| Marker | Meaning |
|---|---|
| `collections_nba` | finvet dataset id and directory name |
| `data/collections_nba` | default local partner tree |
| `collections/nba` | alternate path spelling |
| `finvet.data.collections_nba` | partner loader import |

## When eval loaders land

Future eval modules ([#58](https://github.com/Alberto-Codes/typevet/issues/58),
[#59](https://github.com/Alberto-Codes/typevet/issues/59)) must read **public**
Hub or checked-in fixtures only. Wire the same guard in packaging tests so a
manifest or `pyproject` include cannot regress partner paths.
