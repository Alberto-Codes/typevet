"""Private framing checks and adapter wiring for llama.cpp judgment (#444).

``open_gemma_native_vision_judgment`` calls these helpers so the open function
stays under the LOC gate. Framing is allowed only with
``require_vision=False`` and a served ``DEGRADED_CHATML`` family. The adapter
takes ``framing`` or ``served_template``, not both. An omitted ``n_vocab``
resolves to a catalog lookup when framing is set, else the Gemma class size.

Examples:
    ```python
    from typevet.adapters.outbound.gemma import ServedTemplateClass
    from typevet.adapters.outbound.llama_cpp.judgment_factory_options import (
        check_framing_served_family,
    )

    check_framing_served_family(ServedTemplateClass.DEGRADED_CHATML)
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]:
      Factory that calls these helpers
    - [typevet.adapters.outbound.judgment_scoring][]: ``ScoringJudgmentAdapter``
    - [typevet.ports.framing][]: ``ModelFramingPort``

Attributes:
    ScoringWireInputs (type): Inputs for ``build_scoring_judgment_port``.
    OmittedNVocab (type): Sentinel type for an omitted ``n_vocab`` keyword.
    OMITTED_N_VOCAB (OmittedNVocab): Shared omitted-``n_vocab`` sentinel.
    resolve_factory_n_vocab (Callable): Resolve omitted vs explicit ``n_vocab``.
    check_framing_requires_no_vision (Callable): Refuse framing when vision is
        still required.
    check_framing_served_family (Callable): Refuse framing unless the served
        family is degraded ChatML.
    build_scoring_judgment_port (Callable): Wire scoring, request ids and the
        judgment port.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx

from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp.multimodal import MediaCapability
from typevet.adapters.outbound.llama_cpp.scoring import (
    DEFAULT_N_VOCAB,
    LlamaCppCandidateScoringAdapter,
)
from typevet.adapters.outbound.request_ids import (
    RequestIdJudgmentPort,
    RequestIdScoringPort,
)
from typevet.domain.text_parts import TextParts
from typevet.ports.framing import ModelFramingPort
from typevet.ports.judgment import JudgmentPort
from typevet.ports.scoring import CandidateScoringPort

__all__ = [
    "OMITTED_N_VOCAB",
    "OmittedNVocab",
    "ScoringWireInputs",
    "build_scoring_judgment_port",
    "check_framing_requires_no_vision",
    "check_framing_served_family",
    "resolve_factory_n_vocab",
]


class OmittedNVocab:
    """Sentinel type for an omitted factory ``n_vocab`` keyword.

    Examples:
        ```python
        from typevet.adapters.outbound.llama_cpp.judgment_factory_options import (
            OMITTED_N_VOCAB,
            OmittedNVocab,
        )

        assert isinstance(OMITTED_N_VOCAB, OmittedNVocab)
        ```
    """


OMITTED_N_VOCAB = OmittedNVocab()


def resolve_factory_n_vocab(
    n_vocab: int | OmittedNVocab | None,
    *,
    framing: ModelFramingPort | None,
) -> int | None:
    """Resolve factory ``n_vocab`` for framing vs the Gemma default.

    An omitted value becomes ``None`` (catalog lookup) when ``framing`` is set,
    else ``DEFAULT_N_VOCAB``. An explicit ``int`` or ``None`` wins unchanged.

    Args:
        n_vocab: Explicit size, explicit ``None``, or the omitted sentinel.
        framing: Optional framing from the factory keyword.

    Returns:
        The vocabulary size for the scoring adapter, or ``None`` to read the
        catalog after completion.
    """
    if isinstance(n_vocab, OmittedNVocab):
        return None if framing is not None else DEFAULT_N_VOCAB
    return n_vocab


@dataclass(frozen=True, slots=True)
class ScoringWireInputs:
    """Inputs that ``build_scoring_judgment_port`` needs to wire one port.

    Attributes:
        client (httpx.Client): Shared HTTP client for the router.
        base_url (str): Router base URL for the scoring adapter.
        timeout (float): HTTP timeout in seconds.
        model_id (str): Pinned model id for scoring and media capability.
        n_vocab (int | None): Vocabulary size, or ``None`` to read it later.
        capability (MediaCapability): Media capability for ``model_id``.
        tokenize (Callable): Tokenizer hook for control candidates.
        scoring_port_wrapper (Callable | None): Optional scoring wrapper.
        text_parts (TextParts | None): Option block and context templates.
        served (ServedTemplateClass | None): Served family, or ``None`` with
            framing.
        framing (ModelFramingPort | None): Framing, or ``None`` with served.

    Examples:
        ```python
        from typevet.adapters.outbound.llama_cpp.judgment_factory_options import (
            ScoringWireInputs,
        )

        assert ScoringWireInputs.__dataclass_fields__
        ```
    """

    client: httpx.Client
    base_url: str
    timeout: float
    model_id: str
    n_vocab: int | None
    capability: MediaCapability
    tokenize: Callable[[str], Sequence[int]]
    scoring_port_wrapper: Callable[[CandidateScoringPort], CandidateScoringPort] | None
    text_parts: TextParts | None
    served: ServedTemplateClass | None
    framing: ModelFramingPort | None


def check_framing_requires_no_vision(
    framing: ModelFramingPort | None,
    *,
    require_vision: bool,
) -> None:
    """Refuse framing when the caller still requires vision.

    Args:
        framing: Optional model framing from the factory keyword.
        require_vision: Factory vision requirement.

    Raises:
        ValueError: ``framing`` is set and ``require_vision`` is true.
    """
    if framing is not None and require_vision:
        msg = "framing requires require_vision=False"
        raise ValueError(msg)


def check_framing_served_family(family: ServedTemplateClass) -> None:
    """Refuse framing unless the served family is degraded ChatML.

    Args:
        family: Classified served template family from ``/apply-template``.

    Raises:
        ValueError: ``family`` is not ``DEGRADED_CHATML``.
    """
    if family is not ServedTemplateClass.DEGRADED_CHATML:
        msg = f"framing requires degraded_chatml served template, got {family.value}"
        raise ValueError(msg)


def build_scoring_judgment_port(
    inputs: ScoringWireInputs,
) -> tuple[JudgmentPort, LlamaCppCandidateScoringAdapter]:
    """Build the scoring adapter, request-id wrappers and judgment port.

    Pass ``framing`` or ``served`` on ``inputs``, not both. The factory still
    stores the probed family on the session when the adapter takes framing.

    Args:
        inputs: Client, model, capability, tokenize hook and framing or served
            family.

    Returns:
        The judgment port and the scoring adapter the caller must close.

    Raises:
        ValueError: Both ``framing`` and ``served`` are set on ``inputs``.
    """
    scoring = LlamaCppCandidateScoringAdapter(
        base_url=inputs.base_url,
        timeout=inputs.timeout,
        client=inputs.client,
        n_vocab=inputs.n_vocab,
        media_capabilities={inputs.model_id: inputs.capability},
    )
    scoring_port: CandidateScoringPort = scoring
    if inputs.scoring_port_wrapper is not None:
        scoring_port = inputs.scoring_port_wrapper(scoring)
    adapter = ScoringJudgmentAdapter(
        RequestIdScoringPort(scoring_port),
        tokenize_content=inputs.tokenize,
        served_template=inputs.served,
        framing=inputs.framing,
        pinned_model=inputs.model_id,
        text_parts=inputs.text_parts,
    )
    return RequestIdJudgmentPort(adapter), scoring
