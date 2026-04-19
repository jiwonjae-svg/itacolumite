"""Gemini API client – multimodal content generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from itacolumite.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Accumulated Gemini API usage for the current session."""

    total_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def record(self, usage_metadata: Any) -> None:
        """Extract token counts from a Gemini response's usage_metadata."""
        self.total_calls += 1
        if usage_metadata is None:
            return
        self.prompt_tokens += getattr(usage_metadata, "prompt_token_count", 0) or 0
        self.completion_tokens += getattr(usage_metadata, "candidates_token_count", 0) or 0
        self.total_tokens += getattr(usage_metadata, "total_token_count", 0) or 0

    def reset(self) -> None:
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


def is_placeholder_api_key(api_key: str | None) -> bool:
    """Return True when the configured API key is missing or still a template value."""
    if api_key is None:
        return True

    normalized = api_key.strip()
    return normalized == "" or normalized == "your_api_key_here"


class GeminiClient:
    """Wraps the Google GenAI SDK for Gemini API calls."""

    def __init__(self) -> None:
        settings = get_settings().gemini
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model_fast = settings.gemini_model_fast
        self._model_pro = settings.gemini_model_pro
        self._temperature = settings.gemini_temperature
        self._max_output_tokens = settings.gemini_max_tokens
        self._usage = UsageStats()
        logger.info(
            "Gemini client initialized (fast=%s, pro=%s)",
            self._model_fast,
            self._model_pro,
        )

    @property
    def usage(self) -> UsageStats:
        return self._usage

    def _extract_response_text(self, response: Any) -> str | None:
        """Return the best-effort text payload from a Gemini response."""
        try:
            text = getattr(response, "text", None)
        except ValueError:
            text = None

        if isinstance(text, str) and text.strip():
            return text

        fragments: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    fragments.append(part_text)

        if fragments:
            return "".join(fragments)

        return text if isinstance(text, str) and text else None

    def _response_diagnostics(self, response: Any) -> str:
        """Build a short diagnostic summary for unexpected Gemini responses."""
        details: list[str] = []

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            details.append(f"candidates={len(candidates)}")
            finish_reasons = [
                str(getattr(candidate, "finish_reason", None))
                for candidate in candidates
                if getattr(candidate, "finish_reason", None) is not None
            ]
            if finish_reasons:
                details.append(f"finish_reason={','.join(finish_reasons)}")

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason is not None:
                details.append(f"block_reason={block_reason}")

        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            total_tokens = getattr(usage_metadata, "total_token_count", None)
            if total_tokens is not None:
                details.append(f"total_tokens={total_tokens}")

        return ", ".join(details) if details else "no candidates or prompt feedback"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def generate(
        self,
        contents: list[Any],
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str = "application/json",
        model_override: str | None = None,
    ) -> str:
        """Generate content from Gemini with structured JSON output.

        Args:
            contents: List of content parts (text strings, PIL Images, bytes).
            system_instruction: System prompt.
            temperature: Sampling temperature.
            max_output_tokens: Max response length.
            response_mime_type: Expected response format.

        Returns:
            Raw response text from the model.
        """
        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._temperature,
            max_output_tokens=max_output_tokens if max_output_tokens is not None else self._max_output_tokens,
            response_mime_type=response_mime_type,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        model = model_override or self._model_fast
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        self._usage.record(getattr(response, "usage_metadata", None))

        text = self._extract_response_text(response)
        if not text:
            raise RuntimeError(
                f"Gemini returned empty response ({self._response_diagnostics(response)})"
            )

        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", "N/A") if candidates else "N/A"

        logger.debug(
            "Gemini response: %d chars, model=%s, finish=%s, tokens=%d",
            len(text),
            model,
            finish_reason,
            self._usage.total_tokens,
        )
        return text

    def generate_with_image(
        self,
        text_prompt: str,
        image_bytes: bytes,
        *,
        system_instruction: str | None = None,
        use_pro: bool = False,
        **kwargs,
    ) -> str:
        """Convenience wrapper for text + screenshot prompts."""
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        model_override = self._model_pro if use_pro else None
        return self.generate(
            contents=[text_prompt, image_part],
            system_instruction=system_instruction,
            model_override=model_override,
            **kwargs,
        )

    def validate_api_access(self) -> None:
        """Perform a small non-retried request to fail fast on bad API configuration."""
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8,
            response_mime_type="text/plain",
        )
        config.system_instruction = "Respond with the single word OK."

        response = self._client.models.generate_content(
            model=self._model_fast,
            contents=["ping"],
            config=config,
        )

        text = self._extract_response_text(response)
        if text:
            logger.info("Gemini API validation succeeded (model=%s)", self._model_fast)
            return

        if getattr(response, "candidates", None):
            logger.info(
                "Gemini API validation returned no text but did return candidate data (%s)",
                self._response_diagnostics(response),
            )
            logger.info("Gemini API validation succeeded (model=%s)", self._model_fast)
            return

        raise RuntimeError(
            "Gemini API validation returned an empty response "
            f"({self._response_diagnostics(response)})"
        )
