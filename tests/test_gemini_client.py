"""Tests for Gemini client response handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from itacolumite.ai import gemini_client as gemini_client_module


class _FakeModels:
    def __init__(self, response: object) -> None:
        self._response = response

    def generate_content(self, **_kwargs: object) -> object:
        return self._response


def _make_client(monkeypatch: pytest.MonkeyPatch, response: object) -> gemini_client_module.GeminiClient:
    fake_settings = SimpleNamespace(
        gemini=SimpleNamespace(
            gemini_api_key="valid-test-key",
            gemini_model_fast="gemini-2.5-flash",
            gemini_model_pro="gemini-2.5-pro",
            gemini_temperature=0.3,
            gemini_max_tokens=4096,
        )
    )

    monkeypatch.setattr(gemini_client_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        gemini_client_module.genai,
        "Client",
        lambda api_key: SimpleNamespace(models=_FakeModels(response)),
    )
    return gemini_client_module.GeminiClient()


def test_validate_api_access_accepts_candidate_only_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        text=None,
        candidates=[SimpleNamespace(finish_reason="STOP", content=None)],
        prompt_feedback=None,
        usage_metadata=None,
    )

    client = _make_client(monkeypatch, response)

    client.validate_api_access()


def test_validate_api_access_reports_response_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        text=None,
        candidates=None,
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        usage_metadata=None,
    )

    client = _make_client(monkeypatch, response)

    with pytest.raises(RuntimeError, match="block_reason=SAFETY"):
        client.validate_api_access()


def test_generate_falls_back_to_candidate_part_text(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(parts=[SimpleNamespace(text='{"ok": true}')]),
            )
        ],
        prompt_feedback=None,
        usage_metadata=None,
    )

    client = _make_client(monkeypatch, response)

    assert client.generate(contents=["ping"]) == '{"ok": true}'