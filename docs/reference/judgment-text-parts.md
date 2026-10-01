# Judgment text parts

Kind: reference. This page lists each text part of a typevet judgment call and its evolution status.

A typevet judgment call sends one scoring prefix to the model.
Seven text parts make up that prefix.
The table gives the source location of each part as `path:line` at the time of writing.
Other pages and candidate mappings use the part names in [component names](#component-names).
The parent plan is [#360](https://github.com/Alberto-Codes/typevet/issues/360).

## Parts

| Part | Defined at | Caller-substitutable today | Pinned in receipts | Evolved so far | May ever be evolved |
|---|---|---|---|---|---|
| Question `instructions` | `src/typevet/domain/judgment_questions.py:67` | Yes. The caller sets it on each `Noul`, `Choice` or `Score`. | Wording receipts store seed and evolved text verbatim, with no digest. Identity receipts hash it in `prompt_digests`. | Yes, the DIFrauD `is_scam` Noul ([#309](https://github.com/Alberto-Codes/typevet/issues/309), [#252](https://github.com/Alberto-Codes/typevet/issues/252)). | Yes. |
| `criteria` (true/false text, option labels, scale anchors) | `src/typevet/domain/judgment_questions.py:68` | Yes. The caller sets it on each question. | Wording receipts do not store it. Identity receipts hash it in `prompt_digests`. | No. `WordingTransport` always reuses the seed `criteria`. | Yes, planned ([#363](https://github.com/Alberto-Codes/typevet/issues/363)). |
| Rendered option block | `src/typevet/domain/field_instructions.py:84` | No. typevet renders it from the question. | Only through the commit. | No. | Yes, planned ([#364](https://github.com/Alberto-Codes/typevet/issues/364)). |
| Context / user-text template | `src/typevet/adapters/outbound/judgment_scoring.py:191` | Partly. A framing places the rendered context, but cannot change how typevet renders `state`. | Only through the commit. | No. | Yes, planned ([#364](https://github.com/Alberto-Codes/typevet/issues/364)). |
| Framing preamble | `src/typevet/adapters/outbound/gemma/scoring_prefix.py:97` | Yes, through `ScoringJudgmentAdapter(framing=...)` with a `ModelFramingPort`. | Identity receipts record the served-template family. They do not record a framing class. | No. | Yes. No child issue exists yet. |
| No-thinking prefill | `src/typevet/adapters/outbound/gemma/served_template.py:27` | No. A Gemma 4 framing must end with it, and typevet refuses a framing without it. | Only through the commit and the served-template family. | No. | Never. |
| Control-token rule | `src/typevet/domain/judgment_normalize.py:41` | No. | Only through the commit. | No. | Never. |

## Row notes

- **Question `instructions`.** `Noul` sets it at line 67, `Choice` at line 110 and `Score` at line 151 of `judgment_questions.py`.
- **`criteria`.** `Choice` sets it at line 109 and `Score` at line 150. `_field_criteria` at `judgment_scoring.py:172` turns it into option descriptions.
- **Rendered option block.** `render_field_instructions` writes the question line, the `Options:` list and the `<i> → <label>` lines.
- **Rendered option block.** `_answer_instruction` at `field_instructions.py:75` writes the closing answer rule.
- **Context / user-text template.** `_state_context` at `judgment_scoring.py:185` writes JSON for a non-string `state`.
- **Context / user-text template.** `_media_context` puts one media marker per image before the context.
- **Framing preamble.** The turn markers and the role headers come from `served_template.py`, lines 24 to 36.
- **Framing preamble.** `compose_media_scoring_prefix` at `scoring_prefix.py:113` writes the native Gemma turns.
- **Framing preamble.** `ModelFramingPort` at `src/typevet/ports/framing.py:40` is the caller hook.
- **No-thinking prefill.** `require_no_thinking_prefill` at `scoring_prefix.py:57` refuses a Gemma 4 framing that lacks it ([#354](https://github.com/Alberto-Codes/typevet/issues/354)).
- **No-thinking prefill.** On vLLM, the scoring adapter sends `enable_thinking: false` in place of the prefill text.
- **Control-token rule.** `control_binding_pairs` at line 185 maps labels to the controls `0` to `9`, then `A` to `Z`.
- **Control-token rule.** `bind_control_candidates` at line 217 requires one token per control.

The probability read depends on the prefill and on the control-token rule.
Thus no evolution run may change either part.

## Receipt evidence

- The held-out wording receipt stores `seed_text` and `evolved_text` at `evals/src/typevet_evals/wording/held_out.py:413`.
- The comparison wording receipt stores the same two fields at `evals/src/typevet_evals/wording/comparison.py:218`.
- `prompt_digest` at `evals/src/typevet_evals/experiment_identity.py:227` hashes `instructions`, `criteria` and the label order.
- `RuntimeBuild` records `served_template` at `experiment_identity.py:147`. Its `template_identity` defaults to `unknown`.
- `WordingTransport` builds each question from the evolved text and the seed `criteria` at `evals/src/typevet_evals/wording/transport.py:211`.
- [Judgment live receipts](judgment-live-receipts.md) pin the commit, the model and the template family.

## Component names

A candidate mapping names each part with one snake_case key.
Use these keys only.
gepa-adk accepts only keys that are Python identifiers, so a key never holds a dot.

| Part | Component name | Evolvable |
|---|---|---|
| Question `instructions` | `instructions` | Yes |
| `criteria` of a `Noul` | `criteria_true`, `criteria_false` | Yes |
| `criteria` of a `Choice` or `Score` | Not named yet. A follow-up issue under #360 names them. | Yes |
| Rendered option block | `option_block` | Yes |
| Context / user-text template | `context_template` | Yes |
| Framing preamble | `framing_preamble` | Yes |
| No-thinking prefill | `no_thinking_prefill` | Never |
| Control-token rule | `control_token_rule` | Never |

A mapping that names `no_thinking_prefill` or `control_token_rule` is out of scope for evolution.

## See also

- [Native typed judgments](../explanation/native-typed-judgments.md)
- [Wording evolution on DIFrauD scam messages](../explanation/wording-evolution-difraud.md)
- [Use typevet as a judgevet provider](../how-to/use-typevet-as-a-judgevet-provider.md)
- [Judgment live receipts](judgment-live-receipts.md)
- [Glossary](glossary.md)
