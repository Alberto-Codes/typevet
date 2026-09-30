"""The #252 Jev vs Gemma held-out comparison receipt (#329).

Each judge scores its seed wording and its own evolved wording once each on
the #307 held-out rows. ``evolved_text_for`` takes the evolved wording from
the evolution artifact of the backend's judge: Jev's artifact for ``jev``,
Gemma's artifact for ``llama_cpp`` and ``vllm``. ``comparison_receipt`` gives
per arm Cohen's kappa against the DIFrauD labels, the Brier score, the ECE
and the accuracy, the paired bootstrap intervals (context only) and each
call's latency and input tokens.

The receipt has no pass rule. It does not apply the #309 verdict, and it
makes no claim that one judge can replace the other. ``ValidationRows``
lets a smoke run score validation rows through the same loop without a
held-out row.

Attributes:
    COMPARISON_BACKENDS (tuple[str, ...]): The judges' backends.
    EVOLUTION_PROVIDER (Mapping[str, str]): The evolution artifact's
        ``judge_provider`` each backend needs.
    COMPARISON_SPLITS (tuple[str, ...]): The splits a receipt may name.

Examples:
    ```python
    from typevet_evals.wording.comparison import (
        comparison_receipt,
        evolved_text_for,
    )

    text = evolved_text_for(artifact, backend="jev", seed_text=seed_text)
    subject = ComparisonSubject("jev", "jev", "jev-latest", "test")
    receipt = comparison_receipt(
        run, subject, seed_text=seed_text, evolved_text=text, pins={}, identity={}
    )
    ```

See Also:
    - [typevet_evals.wording.held_out][]: the scoring loop and row guards
    - [typevet_evals.wording.metrics][]: kappa, Brier, ECE and the bootstrap
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from typevet_evals.wording.calls import call_summary
from typevet_evals.wording.held_out import (
    HELD_OUT_SPLIT,
    VALIDATION_SPLIT,
    HeldOutRun,
    ValidationRows,
)
from typevet_evals.wording.metrics import (
    ECE_BINS,
    POSITIVE_THRESHOLD,
    cohen_kappa,
    paired_bootstrap,
    wording_metrics,
)

__all__ = [
    "COMPARISON_BACKENDS",
    "COMPARISON_SPLITS",
    "EVOLUTION_PROVIDER",
    "ComparisonSubject",
    "ValidationRows",
    "comparison_receipt",
    "evolved_text_for",
]

COMPARISON_BACKENDS: Final[tuple[str, ...]] = ("jev", "llama_cpp", "vllm")
EVOLUTION_PROVIDER: Final[Mapping[str, str]] = MappingProxyType(
    {"jev": "jev", "llama_cpp": "gemma", "vllm": "gemma"}
)
COMPARISON_SPLITS: Final[tuple[str, ...]] = (HELD_OUT_SPLIT, VALIDATION_SPLIT)


def evolved_text_for(
    artifact: Mapping[str, Any], *, backend: str, seed_text: str
) -> str:
    """Return the evolved wording of the backend's own judge.

    Args:
        artifact: A #328 evolution artifact.
        backend: ``jev``, ``llama_cpp`` or ``vllm``.
        seed_text: The seed wording the artifact must start from.

    Returns:
        The artifact's ``evolved_text``.

    Raises:
        ValueError: When the backend is unknown, the artifact's
            ``judge_provider`` is not the backend's judge, the artifact is not
            valid or has budget refusals, its seed differs, or its evolved
            wording is the seed.
    """
    if backend not in EVOLUTION_PROVIDER:
        msg = f"backend {backend!r} is not one of {COMPARISON_BACKENDS}"
        raise ValueError(msg)
    expected = EVOLUTION_PROVIDER[backend]
    provider = artifact.get("judge_provider")
    if provider != expected:
        msg = f"backend {backend!r} needs judge_provider {expected!r}, not {provider!r}"
        raise ValueError(msg)
    if artifact.get("valid") is not True:
        msg = "the evolution artifact is not valid"
        raise ValueError(msg)
    if artifact.get("budget_refusals") != 0:
        msg = f"the evolution artifact has budget_refusals {artifact.get('budget_refusals')!r}"
        raise ValueError(msg)
    if artifact.get("seed_text") != seed_text:
        msg = "the evolution artifact's seed is not the is_scam seed"
        raise ValueError(msg)
    evolved = str(artifact["evolved_text"])
    if evolved == seed_text:
        msg = "the evolved wording is the same as the seed"
        raise ValueError(msg)
    return evolved


def _arm(probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    return {
        **wording_metrics(probabilities, labels).to_mapping(),
        "kappa": cohen_kappa(probabilities, labels),
    }


@dataclass(frozen=True, slots=True)
class ComparisonSubject:
    """Which judge a comparison receipt measures, and on which rows.

    Attributes:
        judge (str): The judge label, for example ``gemma_llama_cpp``.
        backend (str): ``jev``, ``llama_cpp`` or ``vllm``.
        model (str): The requested or served model id.
        split (str): ``test`` for the held-out rows, ``validation`` for a smoke.

    Examples:
        ```python
        ComparisonSubject("jev", "jev", "jev-latest", "test")
        ```
    """

    judge: str
    backend: str
    model: str
    split: str

    def __post_init__(self) -> None:
        """Refuse an unknown backend or split.

        Raises:
            ValueError: When ``backend`` or ``split`` is not a known value.
        """
        if self.backend not in COMPARISON_BACKENDS:
            msg = f"backend {self.backend!r} is not one of {COMPARISON_BACKENDS}"
            raise ValueError(msg)
        if self.split not in COMPARISON_SPLITS:
            msg = f"split {self.split!r} is not one of {COMPARISON_SPLITS}"
            raise ValueError(msg)


def comparison_receipt(
    run: HeldOutRun,
    subject: ComparisonSubject,
    *,
    seed_text: str,
    evolved_text: str,
    pins: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, Any]:
    """Build the key-free comparison receipt of one judge.

    Metrics and bootstrap are None when the run stopped or scored no row.
    The receipt's ``split`` is ``run.split``, which ``score_held_out`` takes
    from the row type (#339).

    Args:
        run: The scored run.
        subject: The judge, backend, model and split.
        seed_text: The seed wording.
        evolved_text: The judge's evolved wording.
        pins: Dataset, server, template and weights pins.
        identity: The experiment identity mapping.

    Returns:
        A JSON-serializable receipt.

    Raises:
        ValueError: When ``subject.split`` is not ``run.split``.
    """
    if subject.split != run.split:
        msg = (
            f"subject split {subject.split!r} disagrees with the split "
            f"{run.split!r} of the scored rows"
        )
        raise ValueError(msg)
    labels = [p.label for p in run.pairs]
    seed_p = [p.seed_probability for p in run.pairs]
    evolved_p = [p.evolved_probability for p in run.pairs]
    metrics = bootstrap = None
    if run.stopped is None and run.pairs:
        metrics = {"seed": _arm(seed_p, labels), "evolved": _arm(evolved_p, labels)}
        bootstrap = paired_bootstrap(seed_p, evolved_p, labels).to_mapping()
    return {
        "issue": 329,
        "parent_issue": 252,
        "comparison": "252 Jev vs Gemma; numbers per judge; no substitution claim",
        "pass_rule": None,
        "judge": subject.judge,
        "backend": subject.backend,
        "model": subject.model,
        "split": run.split,
        "positive_threshold": POSITIVE_THRESHOLD,
        "ece_bins": ECE_BINS,
        "seed_text": seed_text,
        "evolved_text": evolved_text,
        "rows": len(run.pairs),
        "calls": {
            "seed": run.seed_calls,
            "evolved": run.evolved_calls,
            "total": run.calls,
        },
        "stopped": run.stopped,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "pairs": [
            {
                "record_id": p.record_id,
                "label": p.label,
                "seed": p.seed_probability,
                "evolved": p.evolved_probability,
            }
            for p in run.pairs
        ],
        "per_call": [r.to_mapping() for r in run.call_records],
        "call_summary": call_summary(run.call_records),
        "pins": dict(pins),
        "identity": dict(identity),
    }
