# Eval complementary manifest (JevBench + open text sets)

Kind: reference.

This page documents the durable inventory filed under
[`evals/complementary-manifest.yaml`](../../evals/complementary-manifest.yaml).
It implements the accepted spec on
[#63](https://github.com/Alberto-Codes/typevet/issues/63) from the
[#54](https://github.com/Alberto-Codes/typevet/issues/54) research ranks.
Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51).

## Primary bench — JevBench

JevBench stays the **primary** typed-decision evaluation path for typevet
(law: [#24](https://github.com/Alberto-Codes/typevet/issues/24),
[TypeLLM, Jev and judgevet](../explanation/typellm-and-judgevet.md)).
The published protocol uses 231 tasks (139 Choice, 74 Noul, 18 Score) without
permutation averaging. Upstream lives in
[TypeLLM/TypeLLM](https://github.com/TypeLLM/TypeLLM).

Complementary sets below are **stress and regression only**. They do not
replace JevBench task mix or Score coverage for v1.

## Ranked complementary sets (machine source of truth)

The YAML file lists ranks 1–7 with license, Decision primitive (Noul / Choice),
split notes, `sample_size: TBD`, loader status, and blockers. Defaults from
accepted #54:

- Prefer permissive licenses; **sample** large corpora.
- Prioritize **primitive gaps** (especially Noul) over topic-matching JevBench.
- Require a **pair `state` schema** before BoolQ, SNLI, or PubMedQA loaders.
- Coordinate CLINC Choice sharding with Banking77 ([#58](https://github.com/Alberto-Codes/typevet/issues/58))
  — do not invent a second intent map ([#53](https://github.com/Alberto-Codes/typevet/issues/53)).

| Rank | Dataset | License | Decision |
|---:|---|---|---|
| 1 | google/boolq | CC BY-SA 3.0 | Noul |
| 2 | google/civil_comments | CC0 1.0 | Noul (toxicity threshold) |
| 3 | qiaojin/PubMedQA (pqa_labeled) | MIT | Choice {yes, no, maybe} |
| 4 | SemEval hyperpartisan | CC BY 4.0 | Noul |
| 5 | clinc/clinc_oos | CC BY 3.0 | Choice (domain shards ≤24) |
| 6 | go_emotions | Apache 2.0 | Choice (single-label policy) |
| 7 | stanfordnlp/snli | CC BY-SA 4.0 | Choice 3-way |

Edit ranks in the YAML when research accepts a new revision; bump
`manifest_version`.

## Out of this manifest (cross-linked)

| Scope | Issue | Where |
|---|---|---|
| finvet public sets (Banking77, DIFrauD, CFPB, ABCD, UCI SMS, …) | [#53](https://github.com/Alberto-Codes/typevet/issues/53) | [Eval partner data policy](eval-partner-data-policy.md) |
| PSAI computer-use (multimodal deferred) | [#52](https://github.com/Alberto-Codes/typevet/issues/52) | YAML `out_of_manifest.psai_computer_use` |
| collections NBA partner data | [#61](https://github.com/Alberto-Codes/typevet/issues/61) | [Eval partner data policy](eval-partner-data-policy.md) |
| OSWorld-class GUI | — | Deferred with PSAI; not #54-ranked |

## Non-goals

- Downloading or vendoring corpora from this repo.
- Implementing loaders (tracked on [#58](https://github.com/Alberto-Codes/typevet/issues/58),
  [#59](https://github.com/Alberto-Codes/typevet/issues/59), and research children from #54).

## Open questions

Sample sizes, share-alike derivatives, and state JSON conventions remain
**TBD** in YAML `open_questions` until child issues close them.
