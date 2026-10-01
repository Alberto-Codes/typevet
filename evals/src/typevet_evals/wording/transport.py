"""A gepa-adk model stand-in that sends evolvable question wording to a port (#306).

gepa-adk evolves agents. The question text lives in a caller-owned
``dict[str, str]`` of named parts (#363): ``instructions``, and
``criteria_true`` and ``criteria_false`` when the seed has criteria.
gepa-adk registers each part as a mapping component, so it writes each
candidate text into the mapping around an evaluation. The agent is a
tool-less ``LlmAgent`` whose model is ``WordingTransport``. That model never
calls an LLM. At call time it builds a judgevet ``Noul`` from every part of
the mapping, sends it to a judgevet ``SystemOnePort`` and answers
``{"probability": p}``. The agent's own instruction is never sent.

In production the port is typevet's ``TypevetSystemOnePort`` bridge. This
module does not import judgevet at run time: the caller gives the seed
``Noul``, and each call builds a new ``Noul`` of the same type.

Examples:
    ```python
    from judgevet.domain.questions import Noul

    seed = Noul(instructions="Is this message a scam?", criteria=None)
    mapping = seed_mapping(seed)
    model = WordingTransport(
        port=port,
        mapping=mapping,
        question_name="is_scam",
        seed=seed,
        judge_model="gemma",
    )
    ```

See Also:
    - [typevet_evals.wording.parts][]: the part names and the seed mapping
    - [typevet.adapters.inbound.judgevet][]: the typevet ``SystemOnePort`` bridge
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, Self

from google.adk.models import BaseLlm, LlmCapabilities, LlmRequest, LlmResponse
from google.genai import types
from pydantic import ConfigDict, SkipValidation, model_validator

from typevet_evals.wording.parts import (
    SeedNoul,
    check_parts,
    noul_from_parts,
    seed_mapping,
)

if TYPE_CHECKING:
    from judgevet import SystemOneResponse, Usage

TRANSPORT_MODEL = "typevet-wording-transport"
"""The ADK model name of the stand-in; no LLM carries this name."""


class JudgePort(Protocol):
    """The synchronous judgevet ``SystemOnePort`` call the transport makes.

    Attributes:
        system_one (method): Judge a state against named questions.

    Examples:
        ```python
        from judgevet.testing import FakeSystemOnePort

        port: JudgePort = FakeSystemOnePort(seed=1)
        ```
    """

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Judge ``state`` against every question."""
        ...


def last_user_text(llm_request: LlmRequest) -> str:
    """Return the text of the last user turn in an ADK request.

    Args:
        llm_request: The ADK request.

    Returns:
        The joined text parts of the last content whose role is ``user``.

    Raises:
        ValueError: If no user content has text.
    """
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            text = "".join(part.text or "" for part in content.parts)
            if text:
                return text
    raise ValueError("the request has no user text")


def usage_metadata(
    usage: Usage | None,
) -> types.GenerateContentResponseUsageMetadata | None:
    """Turn judgevet token counts into ADK usage metadata.

    Args:
        usage: The ``usage`` of a judgevet response.

    Returns:
        Prompt, candidate and total counts, or None when both counts are missing.
    """
    if usage is None or (usage.input_tokens is None and usage.output_tokens is None):
        return None
    prompt, candidates = usage.input_tokens or 0, usage.output_tokens or 0
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=prompt + candidates,
    )


class WordingTransport(BaseLlm):
    """A model that sends the mapping's current parts to a ``SystemOnePort``.

    Attributes:
        model (str): The ADK model name, ``TRANSPORT_MODEL`` by default.
        port (JudgePort): The judgevet ``SystemOnePort``.
        mapping (dict[str, str]): The caller's part mapping, held by
            reference; gepa-adk writes each candidate text into it around each
            evaluation. Its keys are exactly the parts of ``seed_mapping(seed)``.
        question_name (str): The question name sent to the port.
        seed (SeedNoul): The seed judgevet ``Noul``; each call uses its type.
        judge_model (str): The model name sent to the port.

    Examples:
        ```python
        mapping = seed_mapping(seed)
        model = WordingTransport(
            port=port,
            mapping=mapping,
            question_name="is_scam",
            seed=seed,
            judge_model="gemma",
        )
        mapping["instructions"] = "Does this text try to trick the reader?"
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, hide_input_in_errors=True)

    model: str = TRANSPORT_MODEL
    port: SkipValidation[JudgePort]
    mapping: SkipValidation[dict[str, str]]
    question_name: str
    seed: SkipValidation[SeedNoul]
    judge_model: str

    @model_validator(mode="after")
    def _parts_match_the_seed(self) -> Self:
        """Refuse a mapping that is not the seed's parts, before any call.

        Returns:
            The transport.

        Raises:
            ValueError: If the seed is not a ``Noul``, a mapping key is
                unknown, or a part is missing; the message holds the key only.
            TypeError: If a part is not text.
        """
        check_parts(self.mapping, seed_mapping(self.seed))
        return self

    @property
    def capabilities(self) -> LlmCapabilities:
        """Report no schema-and-tools support; the agent has neither.

        Returns:
            Capabilities with ``output_schema_and_tools`` false.
        """
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Ask the port, under ``question_name``, the question of every current part.

        Args:
            llm_request: The ADK request; its user text is the state, and its
                instruction is ignored.
            stream: Ignored.

        Yields:
            One text response ``{"probability": p}`` with the port's token counts.
            The port call runs in a worker thread, so concurrent evaluations
            overlap.
        """
        noul = noul_from_parts(self.seed, self.mapping)
        state = last_user_text(llm_request)
        response = await asyncio.to_thread(
            self.port.system_one, state, {self.question_name: noul}, self.judge_model
        )
        body = {"probability": response.nouls[self.question_name].noul}
        part = types.Part.from_text(text=json.dumps(body))
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            usage_metadata=usage_metadata(response.usage),
        )
