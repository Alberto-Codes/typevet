"""Contract: caller text cannot inject a media marker (#433).

A git diff, a log or a fetched page may hold the literal ``<__media__>``.
The judgment adapter neutralizes that sequence in the state, the
instructions and the criteria before it adds one real marker per image. A
text-only judgment then sends no marker, and a media judgment binds exactly
its real images.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_judgment_scoring_marker_injection.py
    ```

See Also:
    - [typevet.domain.media][]: ``neutralize_media_markers``
    - [typevet.adapters.outbound.judgment_scoring][]: The adapter under test
"""

from __future__ import annotations

import math

import pytest
from judgevet.domain.questions import Noul as JevNoul

from tests.fixtures.judgevet_bridge import FAKE_MODEL, judgment_port
from tests.fixtures.judgment_scoring_contract import adapter_for
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import ScoringValidationError
from typevet.domain.judgment_questions import Choice, Noul
from typevet.domain.media import MEDIA_MARKER, ImageInput

pytestmark = pytest.mark.contract

_DIFF = f"+    marker = {MEDIA_MARKER!r}\n-    marker = '<<__media__>__media__>'\n"
_NOUL_LOGPROBS = {"True": math.log(0.6), "False": math.log(0.4)}
_CHOICE_LOGPROBS = {"billing": math.log(0.7), "technical": math.log(0.3)}


def test_text_state_with_marker_judges_without_an_image() -> None:
    adapter, fake = adapter_for(logprobs_by_call=[_NOUL_LOGPROBS])
    response = adapter.judge(_DIFF, {"flagged": Noul()}, "text-model")
    assert "flagged" in response.answers
    assert fake.calls[0].media == ()
    assert fake.calls[0].prefix.count(MEDIA_MARKER) == 0
    assert "media" in fake.calls[0].prefix


def test_marker_in_instructions_and_criteria_is_neutralized() -> None:
    adapter, fake = adapter_for(logprobs_by_call=[_NOUL_LOGPROBS, _CHOICE_LOGPROBS])
    questions = {
        "flagged": Noul(instructions=f"Is {MEDIA_MARKER} present?"),
        "route": Choice(
            criteria={"billing": f"Money {MEDIA_MARKER}", "technical": "Bugs"},
            instructions="Pick:",
        ),
    }
    adapter.judge({"diff": _DIFF}, questions, "text-model")
    assert [call.prefix.count(MEDIA_MARKER) for call in fake.calls] == [0, 0]


def test_media_judgment_binds_only_the_real_images() -> None:
    adapter, fake = adapter_for(
        logprobs_by_call=[_NOUL_LOGPROBS],
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    image = ImageInput(data=b"\x89PNG\r\n\x1a\nimage", mime_type="image/png")
    question = Noul(instructions=f"Does {MEDIA_MARKER} match?")
    adapter.judge(_DIFF, {"flagged": question}, "gemma-mm", media=(image,))
    assert fake.calls[0].media == (image,)
    assert fake.calls[0].prefix.count(MEDIA_MARKER) == 1


def test_bridge_judges_a_diff_holding_the_marker() -> None:
    port = TypevetSystemOnePort(judgment_port())
    response = port.system_one(_DIFF, {"flagged": JevNoul()}, FAKE_MODEL)
    assert set(response.answers) == {"flagged"}


def test_marker_count_mismatch_names_counts_not_content() -> None:
    private_line = "private-diff-line"
    with pytest.raises(ScoringValidationError) as caught:
        CandidateScoringRequest(
            model="m",
            prefix=f"{private_line} {MEDIA_MARKER} {MEDIA_MARKER}",
            candidates=(CandidateTokenSpec(label="True", token_ids=(1,)),),
        )
    message = str(caught.value)
    assert "holds 2" in message
    assert "0 image(s)" in message
    assert private_line not in message
