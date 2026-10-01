"""Question wording that gepa-adk evolves (#259).

The wording lives in a caller-owned ``dict[str, str]`` of named parts:
``instructions``, ``criteria_true`` and ``criteria_false`` (#363). A
tool-less ADK agent uses ``WordingTransport`` as its model. That model sends
a judgevet ``Noul`` built from every current part to a judgevet
``SystemOnePort``.
``evolve_wording`` runs gepa-adk on train and validation records with a
Brier reward and a proposal length cap (#308). ``score_held_out`` and
``held_out_receipt`` check the seed and evolved wording on the held-out rows
against the pre-registered pass rule (#309). ``TimedJudgePort`` records each
call's latency and input tokens, and the ``served`` probes require the native
Gemma 4 template (#327). ``comparison_receipt`` records the #252 Jev vs Gemma
comparison with Cohen's kappa and no pass rule (#329).

Attributes:
    ALLOW_DEGRADED_ENV (str): The variable that allows a non-native template.
    TEXT_JUDGE (str): The router alias that serves native Gemma 4 for text.
    BrierScorer (type): The gepa-adk reward, one minus the Brier score.
    CallRecord (type): The latency and input tokens of one judge call.
    ComparisonSubject (type): The judge, backend, model and split of a receipt.
    ValidationRows (type): Validation records for a smoke of the held-out loop.
    cohen_kappa (callable): Cohen's kappa of the readings against the labels.
    comparison_receipt (callable): Build the #252 comparison receipt.
    evolved_text_for (callable): The evolved wording of the backend's judge.
    TimedJudgePort (type): A ``SystemOnePort`` wrapper that records each call.
    call_summary (callable): Total the call records for a receipt.
    probe_llama_template (callable): Classify the llama.cpp served template.
    probe_vllm_template (callable): Classify the vLLM served chat template.
    require_native_template (callable): Refuse a non-native template.
    HeldOutRows (type): Held-out records checked against the excluded ids.
    HeldOutRun (type): The scored held-out pairs and the call count.
    PairedBootstrap (type): Paired bootstrap intervals of the differences.
    PassVerdict (type): The pre-registered pass rule's clauses and verdict.
    ScoredPair (type): Both wordings' answers for one held-out row.
    WordingMetrics (type): Accuracy, Brier score and ECE of one wording.
    WordingRun (type): The seed and evolved wording and gepa-adk's result.
    WordingRunConfig (type): Settings for one wording evolution.
    brier_score (callable): The Brier score of one probability.
    evolve_wording (callable): Evolve the wording on train, select on validation.
    evolution_artifact (callable): Record the seed and evolved wording of a run.
    held_out_receipt (callable): Build the key-free held-out receipt.
    paired_bootstrap (callable): Resample rows and bound the differences.
    pass_verdict (callable): Apply the pre-registered pass rule.
    reflection_prompt (callable): The reflection prompt with the length limit.
    score_held_out (callable): Ask both wordings about each held-out row.
    stratified_subset (callable): Pick a fixed subset with the label shares.
    train_subset (callable): The train rows the evolution reflects on.
    wording_metrics (callable): Measure one wording over its rows.
    length_cap (callable): A proposal validator that caps each part's length.
    part_caps (callable): The length cap of each evolvable part.
    PART_NAMES (tuple[str, ...]): The part names a run may evolve.
    WordingParts (type): The evolved selection and both full part mappings.
    seed_mapping (callable): The full part mapping of a seed ``Noul``.
    TRANSPORT_MODEL (str): The ADK model name of the stand-in.
    JudgePort (type): The synchronous ``SystemOnePort`` call the transport makes.
    SeedNoul (type): The seed judgevet ``Noul`` shape.
    WordingTransport (type): The model that sends the current wording.
    last_user_text (callable): The last user text of an ADK request.
    usage_metadata (callable): judgevet token counts to ADK usage metadata.

Examples:
    ```python
    from typevet_evals.wording import WordingTransport, seed_mapping

    mapping = seed_mapping(seed)
    model = WordingTransport(
        port=port, mapping=mapping, key="is_scam", seed=seed, judge_model="gemma"
    )
    ```

See Also:
    - [typevet_evals.wording.transport][]: the transport module
    - [typevet_evals.wording.parts][]: the part names and the frozen-part check
    - [typevet_evals.wording.runner][]: the evolution runner
    - [typevet_evals.wording.metrics][]: held-out metrics and the pass rule
    - [typevet_evals.wording.held_out][]: held-out scoring and receipts
    - [typevet_evals.wording.comparison][]: the #252 comparison receipt
    - [typevet_evals.wording.calls][]: per-call latency and input tokens
    - [typevet_evals.wording.served][]: served-template probes
    - [typevet.adapters.inbound.judgevet][]: the typevet ``SystemOnePort`` bridge
"""

from __future__ import annotations

from typevet_evals.wording.calls import CallRecord, TimedJudgePort, call_summary
from typevet_evals.wording.comparison import (
    ComparisonSubject,
    comparison_receipt,
    evolved_text_for,
)
from typevet_evals.wording.held_out import (
    HeldOutRows,
    HeldOutRun,
    ScoredPair,
    ValidationRows,
    evolution_artifact,
    held_out_receipt,
    score_held_out,
    stratified_subset,
    train_subset,
)
from typevet_evals.wording.metrics import (
    PairedBootstrap,
    PassVerdict,
    WordingMetrics,
    cohen_kappa,
    paired_bootstrap,
    pass_verdict,
    wording_metrics,
)
from typevet_evals.wording.parts import PART_NAMES, WordingParts, seed_mapping
from typevet_evals.wording.runner import (
    BrierScorer,
    WordingRun,
    WordingRunConfig,
    brier_score,
    evolve_wording,
    length_cap,
    part_caps,
    reflection_prompt,
)
from typevet_evals.wording.served import (
    ALLOW_DEGRADED_ENV,
    TEXT_JUDGE,
    probe_llama_template,
    probe_vllm_template,
    require_native_template,
)
from typevet_evals.wording.transport import (
    TRANSPORT_MODEL,
    JudgePort,
    SeedNoul,
    WordingTransport,
    last_user_text,
    usage_metadata,
)

__all__ = [
    "ALLOW_DEGRADED_ENV",
    "PART_NAMES",
    "TEXT_JUDGE",
    "TRANSPORT_MODEL",
    "BrierScorer",
    "CallRecord",
    "ComparisonSubject",
    "HeldOutRows",
    "HeldOutRun",
    "JudgePort",
    "PairedBootstrap",
    "PassVerdict",
    "ScoredPair",
    "SeedNoul",
    "TimedJudgePort",
    "ValidationRows",
    "WordingMetrics",
    "WordingParts",
    "WordingRun",
    "WordingRunConfig",
    "WordingTransport",
    "brier_score",
    "call_summary",
    "cohen_kappa",
    "comparison_receipt",
    "evolution_artifact",
    "evolve_wording",
    "evolved_text_for",
    "held_out_receipt",
    "last_user_text",
    "length_cap",
    "paired_bootstrap",
    "part_caps",
    "pass_verdict",
    "probe_llama_template",
    "probe_vllm_template",
    "reflection_prompt",
    "require_native_template",
    "score_held_out",
    "seed_mapping",
    "stratified_subset",
    "train_subset",
    "usage_metadata",
    "wording_metrics",
]
