"""Contract tests for JudgmentPort offline fakes."""

from __future__ import annotations

import pytest

from tests.fixtures.judgment_contract import exc_type_from_name, fake_for, get_fixtures
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput
from typevet.ports.judgment import JudgmentPort


@pytest.mark.contract
@pytest.mark.parametrize("fixture", get_fixtures(), ids=lambda f: f["name"])
def test_judgment_fake_honors_fixture(fixture: dict) -> None:
    fake = fake_for(fixture)
    expect = fixture["expect"]
    if expect["kind"] == "success":
        response = fake.judge(
            fixture["state"],
            fixture["questions"],
            fixture["model"],
        )
        assert response.model == fixture["model"]
        for name, prob in expect.get("nouls", {}).items():
            assert response.nouls[name].noul == prob
        for name, label in expect.get("choices", {}).items():
            assert response.choices[name].choice == label
        for name, score in expect.get("scores", {}).items():
            assert response.scores[name].score == score
        assert len(fake.calls) == 1
        return

    exc_type = exc_type_from_name(expect["exc_type"])
    with pytest.raises(exc_type):
        fake.judge(fixture["state"], fixture["questions"], fixture["model"])
    assert len(fake.calls) == 1


@pytest.mark.contract
def test_judgment_fake_accepts_keyword_only_media() -> None:
    fake = fake_for({})
    media = (ImageInput(data=b"\x89PNG\r\n\x1a\nbytes", mime_type="image/png"),)

    def _accept(port: JudgmentPort) -> None:
        port.judge("state", {"flagged": Noul()}, "fake-judgment", media=media)

    _accept(fake)
    assert fake.media_calls == [media]


@pytest.mark.contract
def test_judgment_fake_defaults_media_to_empty() -> None:
    fake = fake_for({})
    fake.judge("state", {"flagged": Noul()}, "fake-judgment")
    assert fake.media_calls == [()]
