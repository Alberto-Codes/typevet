"""Question wording that gepa-adk evolves (#259).

The wording lives in a caller-owned ``dict[str, str]``. A tool-less ADK
agent uses ``WordingTransport`` as its model. That model sends the current
wording as a judgevet ``Noul`` to a judgevet ``SystemOnePort``.
``evolve_wording`` runs gepa-adk on train and validation records with a
Brier reward and a proposal length cap (#308). ``score_held_out`` and
``held_out_receipt`` check the seed and evolved wording on the held-out rows
against the pre-registered pass rule (#309).

Attributes:
    BrierScorer (type): The gepa-adk reward, one minus the Brier score.
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
    length_cap (callable): A proposal validator that caps the proposal length.
    TRANSPORT_MODEL (str): The ADK model name of the stand-in.
    JudgePort (type): The synchronous ``SystemOnePort`` call the transport makes.
    SeedNoul (type): The seed judgevet ``Noul`` shape.
    WordingTransport (type): The model that sends the current wording.
    last_user_text (callable): The last user text of an ADK request.
    usage_metadata (callable): judgevet token counts to ADK usage metadata.

Examples:
    ```python
    from typevet_evals.wording import WordingTransport

    model = WordingTransport(
        port=port, mapping=mapping, key="is_scam", seed=seed, judge_model="gemma"
    )
    ```

See Also:
    - [typevet_evals.wording.transport][]: the transport module
    - [typevet_evals.wording.runner][]: the evolution runner
    - [typevet_evals.wording.metrics][]: held-out metrics and the pass rule
    - [typevet_evals.wording.held_out][]: held-out scoring and receipts
    - [typevet.adapters.inbound.judgevet][]: the typevet ``SystemOnePort`` bridge
"""

from __future__ import annotations

from typevet_evals.wording.held_out import (
    HeldOutRows,
    HeldOutRun,
    ScoredPair,
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
    paired_bootstrap,
    pass_verdict,
    wording_metrics,
)
from typevet_evals.wording.runner import (
    BrierScorer,
    WordingRun,
    WordingRunConfig,
    brier_score,
    evolve_wording,
    length_cap,
    reflection_prompt,
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
    "TRANSPORT_MODEL",
    "BrierScorer",
    "HeldOutRows",
    "HeldOutRun",
    "JudgePort",
    "PairedBootstrap",
    "PassVerdict",
    "ScoredPair",
    "SeedNoul",
    "WordingMetrics",
    "WordingRun",
    "WordingRunConfig",
    "WordingTransport",
    "brier_score",
    "evolution_artifact",
    "evolve_wording",
    "held_out_receipt",
    "last_user_text",
    "length_cap",
    "paired_bootstrap",
    "pass_verdict",
    "reflection_prompt",
    "score_held_out",
    "stratified_subset",
    "train_subset",
    "usage_metadata",
    "wording_metrics",
]
