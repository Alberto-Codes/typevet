"""Unit tests for Gemma served-template classification (#118)."""

from __future__ import annotations

import pytest

from tests.fixtures.gemma_template_contract import (
    degraded_chatml_rendered,
    native_gemma4_rendered,
)
from typevet.gemma_served_template import (
    CHATML_IM_END,
    ServedTemplateClass,
    classify_served_template,
    label_embeds_control_fragment,
    stop_markers_for,
)


@pytest.mark.unit
def test_classify_degraded_chatml_exact_enum_string() -> None:
    rendered = degraded_chatml_rendered("Pick one label.")
    assert classify_served_template(rendered) is ServedTemplateClass.DEGRADED_CHATML


@pytest.mark.unit
def test_classify_native_gemma4_turn() -> None:
    rendered = native_gemma4_rendered("Hello.")
    assert classify_served_template(rendered) is ServedTemplateClass.NATIVE_GEMMA4_TURN


@pytest.mark.unit
def test_classify_unsupported_mixed_markers() -> None:
    rendered = degraded_chatml_rendered("x") + "<|turn>"
    assert classify_served_template(rendered) is ServedTemplateClass.UNSUPPORTED


@pytest.mark.unit
def test_classify_unsupported_plain_text() -> None:
    assert classify_served_template("plain prompt") is ServedTemplateClass.UNSUPPORTED


@pytest.mark.unit
def test_stop_markers_for_degraded_include_chatml_end() -> None:
    markers = stop_markers_for(ServedTemplateClass.DEGRADED_CHATML)
    assert CHATML_IM_END in markers
    assert "<turn|>" in markers


@pytest.mark.unit
def test_label_embeds_control_fragment_detects_im_start() -> None:
    assert label_embeds_control_fragment("x<|im_start|>y") is True
    assert label_embeds_control_fragment("anger") is False
