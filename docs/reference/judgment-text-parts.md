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
| Question `instructions` | `src/typevet/domain/judgment_questions.py:67` | Yes. The caller sets it on each `Noul`, `Choice` or `Score`. | Wording receipts store seed and evolved text verbatim and by digest. Identity receipts hash it in `prompt_digests`. | Yes, the DIFrauD `is_scam` Noul ([#309](https://github.com/Alberto-Codes/typevet/issues/309), [#252](https://github.com/Alberto-Codes/typevet/issues/252)). | Yes. |
| `criteria` (true/false text, option labels, scale anchors) | `src/typevet/domain/judgment_questions.py:68` | Yes. The caller sets it on each question. | Wording receipts store the `Noul` criteria in `seed_parts` and `evolved_parts` when a run passes its parts. Identity receipts hash it in `prompt_digests`. | No live run yet. A run can evolve `Noul` criteria since [#363](https://github.com/Alberto-Codes/typevet/issues/363). | Yes. `Choice` and `Score` criteria are not named yet. |
| Rendered option block | `src/typevet/domain/text_parts.py:74` | Yes, as a template ([#364](https://github.com/Alberto-Codes/typevet/issues/364)). The caller sets `TextParts(option_block=...)`. | `JudgmentResponse.text_parts` records its digest, or `"default"` when unset. | No. | Yes. |
| Context / user-text template | `src/typevet/domain/text_parts.py:77` | Yes, as a template ([#364](https://github.com/Alberto-Codes/typevet/issues/364)). The caller sets `TextParts(context_template=...)`. A framing gets the rendered user text ([#373](https://github.com/Alberto-Codes/typevet/issues/373)). | `JudgmentResponse.text_parts` records its digest, or `"default"` when unset. | No. | Yes. |
| Framing preamble | `src/typevet/adapters/outbound/gemma/scoring_prefix.py:104` | Yes, through `ScoringJudgmentAdapter(framing=...)` with a `ModelFramingPort`. | Identity receipts record the served-template family. They do not record a framing class. | No. | Yes. No child issue exists yet. |
| No-thinking prefill | `src/typevet/adapters/outbound/gemma/served_template.py:27` | No. A Gemma 4 framing must end with it, and typevet refuses a framing without it. | Only through the commit and the served-template family. | No. | Never. |
| Control-token rule | `src/typevet/domain/judgment_normalize.py:41` | No. | Only through the commit. | No. | Never. |

## Row notes

- **Question `instructions`.** `Noul` sets it at line 67, `Choice` at line 110 and `Score` at line 151 of `judgment_questions.py`.
- **`criteria`.** `Choice` sets it at line 109 and `Score` at line 150. `_field_criteria` at `judgment_scoring.py:170` turns it into option descriptions.
- **Rendered option block.** `render_field_instructions` at `field_instructions.py:109` renders the `option_block` template.
- **Rendered option block.** `_answer_instruction` at `field_instructions.py:72` writes the answer rule for the `{answer_rule}` placeholder.
- **Context / user-text template.** `render_context` at `text_parts.py:245` renders the `context_template` template.
- **Context / user-text template.** `_state_context` at `judgment_scoring.py:183` writes JSON for a non-string `state`.
- **Context / user-text template.** `_media_context` puts one media marker per image before the context.
- **Framing preamble.** The turn markers and the role headers come from `served_template.py`, lines 24 to 36.
- **Framing preamble.** `compose_media_scoring_prefix` at `scoring_prefix.py:124` writes the native Gemma turns.
- **Framing preamble.** `ModelFramingPort` at `src/typevet/ports/framing.py:41` is the caller hook.
- **No-thinking prefill.** `require_no_thinking_prefill` at `scoring_prefix.py:64` refuses a Gemma 4 framing that lacks it ([#354](https://github.com/Alberto-Codes/typevet/issues/354)).
- **No-thinking prefill.** On vLLM, the scoring adapter sends `enable_thinking: false` in place of the prefill text.
- **Control-token rule.** `control_binding_pairs` at line 185 maps labels to the controls `0` to `9`, then `A` to `Z`.
- **Control-token rule.** `bind_control_candidates` at line 217 requires one token per control.

The probability read depends on the prefill and on the control-token rule.
Thus no evolution run may change either part.

## Option block and context templates

`ScoringJudgmentAdapter(..., text_parts=TextParts(...))` takes both templates.
`TextParts` is in `typevet.domain.text_parts`.
An unset part uses its default, and the prefix does not change.
`TextParts` refuses a bad template when the caller builds it, before any scoring call.

| Template | Default (`DEFAULT_OPTION_BLOCK`, `DEFAULT_CONTEXT_TEMPLATE`) | Required placeholders | Optional placeholders |
|---|---|---|---|
| `option_block` | `{name}: {question}\n\nOptions:\n{control} → {label}{description}\n\n{answer_rule}` | `{question}`, `{control}`, `{label}`, `{description}`, `{answer_rule}` | `{name}` |
| `context_template` | `{context}\n\n{field_block}` | `{context}`, `{field_block}` | None |

Placeholder values:

- `{name}` is the question id, and `{question}` is the question `instructions`.
- `{control}` is the control string, with the `Control` word when typevet adds it.
- `{label}` is the option label.
- `{description}` is `": "` and the option criteria text, or empty when the option has no criteria.
- `{answer_rule}` is the answer rule sentence that typevet writes.
  It is required, because the control read is valid only when the model gets the rule ([#373](https://github.com/Alberto-Codes/typevet/issues/373)).
- `{context}` is the state context. It starts with one media marker per image.
- `{field_block}` is the rendered `option_block`.

The line that holds `{control}` is the option line.
typevet writes it once per option, in control order.
`{control}`, `{label}` and `{description}` must be on the option line.
The other placeholders must not be on it.

`TextParts` refuses a template for these rules:

- A required placeholder is missing.
- A placeholder occurs more than once.
- The template has an unknown placeholder, a conversion such as `!r`, or a format spec.
- The option line does not render `<control> → <label>`. This rule is "drops a control line".
- The template has a gold-reference marker from `gold_reference_markers()`.
- The template has the media marker.
- The template has a chat turn marker: `<|im_start|>`, `<|im_end|>`, `<start_of_turn>`, `<end_of_turn>`, `<|turn>` or `<turn|>`.
  `TURN_MARKERS` lists them.
  These are the ChatML, Gemma 3 and Gemma 4 markers from `served_template.py`.
  A turn marker in a template can open or close a turn, so typevet refuses it ([#373](https://github.com/Alberto-Codes/typevet/issues/373)).
- The template has a Gemma 4 control token: `<|channel>`, `<channel|>`, `<|think|>` or `<|tool_response>`.
  `TURN_MARKERS` also lists these tokens, and the rule message calls each one a turn marker.
  These tokens can open a thinking channel or a tool reply ([#373](https://github.com/Alberto-Codes/typevet/issues/373)).

The rules check the templates only.
typevet does not check caller values (state, criteria and labels), with or without text parts.

The error names the part and the rule.
It never holds the template text.
Write a literal brace as `{{` or `}}`.

`ScoringJudgmentAdapter` renders the `context_template` first, then gives the user text to a framing.
`ModelFramingPort.compose_prefix(*, user_text, media)` wraps that text in turn markers.
Thus both text parts apply under every framing ([#373](https://github.com/Alberto-Codes/typevet/issues/373)).
The prefill check still applies to the framing prefix.
`ScoringJudgmentAdapter` refuses a framing together with a `served_template`.
`open_vllm_judgment` and `open_gemma_native_vision_judgment` take an optional `text_parts` keyword.

`TextParts.receipt()` gives the `text_parts` receipt block.
It has one key for each part: `option_block` and `context_template`.
The value is the SHA-256 hex digest of the UTF-8 template text.
The value is `"default"` when the part is unset.
`ScoringJudgmentAdapter.text_parts` gives the parts of one adapter.
The scoring adapter puts this block in `JudgmentResponse.text_parts` on every response.
A response from another adapter has an empty `text_parts`.

## Receipt evidence

- The held-out wording receipt stores its wording keys at `evals/src/typevet_evals/wording/held_out.py:454`.
- The comparison wording receipt stores the same keys at `evals/src/typevet_evals/wording/comparison.py:277`.
- `wording_fields` at `evals/src/typevet_evals/wording/digests.py:138` builds those keys. [Wording parts block](eval-receipt-blocks.md#wording-parts-block) lists them.
- `prompt_digest` at `evals/src/typevet_evals/experiment_identity.py:227` hashes `instructions`, `criteria` and the label order.
- `RuntimeBuild` records `served_template` at `experiment_identity.py:147`. Its `template_identity` defaults to `unknown`.
- `WordingTransport` builds each question from every part of the mapping at `evals/src/typevet_evals/wording/transport.py:199`.
- [Judgment live receipts](judgment-live-receipts.md) pin the commit, the model and the template family.

## Component names

A candidate mapping names each part with one snake_case key.
Use these keys only.
gepa-adk accepts only keys that are Python identifiers, so a key never holds a dot.
A wording run uses the `Noul` keys ([#363](https://github.com/Alberto-Codes/typevet/issues/363)) and the `Choice` keys ([#369](https://github.com/Alberto-Codes/typevet/issues/369)).

| Part | Component name | Evolvable |
|---|---|---|
| Question `instructions` | `instructions` | Yes |
| `criteria` of a `Noul` | `criteria_true`, `criteria_false` | Yes |
| `criteria` of a `Choice`, every label an identifier | `criteria_<label>` | Yes |
| `criteria` of a `Choice`, one label not an identifier | `option_<i>`, the 0-based label position | Yes |
| `criteria` of a `Score` | `level_<i>`, the 0-based level index | Yes |
| Rendered option block | `option_block` | Yes |
| Context / user-text template | `context_template` | Yes |
| Framing preamble | `framing_preamble` | Yes |
| No-thinking prefill | `no_thinking_prefill` | Never |
| Control-token rule | `control_token_rule` | Never |

A mapping that names `no_thinking_prefill` or `control_token_rule` is out of scope for evolution.

A `Choice` uses one name rule for all its options, never a mix.
When every label is an identifier, a `true` or `false` label gives the `Noul` name, such as `criteria_true`.
A `None` description has no part.
A description that is not text and not `None` refuses the seed.
That error names the label or the level, never the text.
`part_table` in `typevet_evals.wording.parts` maps each criterion name to its label or level index.
For a `Noul`, it maps `criteria_true` to `true` and `criteria_false` to `false`.
`question_mapping` builds the full mapping of a `Noul`, a `Choice` or a `Score` ([#369](https://github.com/Alberto-Codes/typevet/issues/369)).
`question_from_parts` builds the question again and keeps the label order.
`part_roles` gives the role of each part for the reflection prompt.

## Wording run parts

`seed_mapping` in `typevet_evals.wording.parts` builds the full mapping from a seed `Noul` or `Choice`.
A seed without `criteria` gives the `instructions` part only.
A `Noul` seed with `criteria` must hold exactly a `true` and a `false` text.
`criteria_true` holds the `true` text, and `criteria_false` holds the `false` text.
A `Choice` seed gives one part per described label, as the table above names it.
The wording run refuses a `Score` seed until a level scorer exists.

The seed type sets the reward and the transport answer.

| Seed | Transport answer | Reward per row | Gold label |
|---|---|---|---|
| `Noul` | `{"probability": p}` | `BrierScorer`: 1 − (p − gold)² | `"1"` or `"0"` |
| `Choice` | `{"probabilities": {label: p}}` | `ChoiceScorer`: 1 − ½ Σ (p_k − 1[k = gold])² | The label name |

A `WordingRow` holds the state, the gold label, the record id and the split.
The state is a message text or a state mapping, such as a PubMedQA question.
The gold label never reaches the port.
`ChoiceScorer` refuses a distribution that lacks a seed label or holds another label.
The held-out receipt of a `Choice` run adds Cohen's kappa to accuracy, Brier and ECE.
Its ECE uses the probability of the chosen label.
The #133 ECE reference applies to `Noul` runs only.
The evolution artifact and the held-out receipt record `part_table` beside `seed_parts`.

`evolve_wording(..., components=[...])` names the parts that evolve.
The default is `["instructions"]`.
gepa-adk evolves the named parts only, and the other parts are frozen.
After the run, a check compares each frozen part with its seed text.
A changed frozen part fails the run.

The transport refuses an unknown key or a missing part.
Its error names the key and never the text.

## See also

- [Native typed judgments](../explanation/native-typed-judgments.md)
- [Wording evolution on DIFrauD scam messages](../explanation/wording-evolution-difraud.md)
- [Use typevet as a judgevet provider](../how-to/use-typevet-as-a-judgevet-provider.md)
- [Judgment live receipts](judgment-live-receipts.md)
- [Glossary](glossary.md)
