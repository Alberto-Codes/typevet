# Verified evidence and inferred claims

Kind: explanation.

A typevet change can pass every gate and still overstate what the repo has
shown. **Verified** means a named check exercised the claim on the inputs it
used. **Inferred** means the claim follows from design, docs or analogy but no
check in this repo has exercised it yet. Do not promote inferred claims to
verified without that exercise.

Sister project judgevet uses the same split for service and policy evidence.
See [What the evidence can establish](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/verification.md)
for fixture labeling and scope limits. typevet applies the same discipline to
local generation on llama.cpp.

## Three layers, three scopes

The [testing pyramid](../reference/glossary.md#terms) is law in
[CLAUDE.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md) and [AGENTS.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md). Default pytest
runs `unit` and `contract` only. It excludes `live`. Coverage on that default
suite must stay at or above 90. A live pass does not replace unit or contract.

| Layer | Where | What a pass can verify | What it cannot verify |
|---|---|---|---|
| **Unit** | `tests/unit/`, marker `unit` | Domain invariants, inbound wiring with fakes, pure logic with controlled inputs. No network. | Real HTTP, real model weights, or whether a remote server matches production. |
| **Contract** | `tests/contract/`, marker `contract` | Fake port and real adapter behave the same on the cases the suite runs (shared fixtures when present). Parsing, validation and error paths for those cases. | Every unseen request shape, live server quirks, or model judgment quality. |
| **Live** | `tests/live/`, marker `live` | One exercised call completed for the prompt, schema and model that test used. Parsed output matched schema constraints for that run. | Statistical calibration, correctness of sentiment or scores, or identical future answers. |

Each layer answers a different question. Unit asks whether local code behaves
as specified when inputs are synthetic. Contract asks whether outbound
adapters honor the same port contract on labeled fixtures. Live asks whether
the configured router and model can complete one real generation path today.

## Fixture labeling (judgevet shape)

Label what a fixture is so readers know the scope of a green test.

**Synthetic fixture.** The test author constructs the input or the transport
response. A fake that returns `{"answer": 42}` verifies validation and port
wiring. It does not show what Gemma 4 would return for a ticket.

**Shared contract fixture.** The same request (and expected shape or error)
runs through the fake generation adapter and the llama.cpp adapter. Agreement
on those cases verifies adapter compatibility for them. When HTTP is mocked,
the case does not verify the live server unless a live test exercises it.

**Live observation.** The test calls the configured base URL and model. A pass
verifies only that exercised prompt and schema. It does not verify every enum
value, every error body, or every model id in the catalog.

Issue [#28](https://github.com/Alberto-Codes/typevet/issues/28) tracks
stronger shared fixtures under `tests/fixtures/` (judgevet shape). Until that
lands, treat each contract test’s inline schema and mock response as its own
labeled fixture set.

## Structure is not judgment

Grammar-JSON and schema validation prove **valid structure** for the cases
run. They do not prove that a field is **correct** for your task. A live test
that asserts `sentiment` is one of `pos`, `neg`, `neu` verifies shape and
enum membership for that run. It does not verify that the model chose the
right sentiment.

The MVP path is documented in [TypeLLM, Jev and judgevet](typellm-and-judgevet.md).
Logprob scoring and live judgment adapters remain design targets until
dedicated tests and live evidence name them verified. The judgment port
surface and answer validators are contract-tested offline ([#101](https://github.com/Alberto-Codes/typevet/issues/101)).

Do not claim expected calibration error (ECE) or other statistical quality
from passing unit, contract or live tests. Those require labeled task
evaluations outside the default pyramid.

## How to write claims on issues and in docs

When you report evidence:

1. Name the layer (`unit`, `contract`, or `live`) and the test or command.
2. State the inputs (fixture label, model id, schema shape) the check used.
3. Separate verified facts from inferred next steps.

When two sources disagree, record both and the open question. A live
observation settles the exercised case. It does not automatically settle every
model or error path.

## Related pages

- [Testing pyramid](../reference/glossary.md#terms) (glossary entry)
- [Testing pyramid law](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md) (non-negotiables table)
- [Run Gemma 4 on llama.cpp](../how-to/run-gemma4-llamacpp.md) (live opt-in path)
- judgevet: [verification](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/verification.md),
  [judgments](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/judgments.md)
