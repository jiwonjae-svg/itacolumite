"""Gemini API client – multimodal content generation."""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from itacolumite.config.settings import get_settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wraps the Google GenAI SDK for Gemini API calls."""

    def __init__(self) -> None:
        settings = get_settings().gemini
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        self._temperature = settings.gemini_temperature
        self._max_output_tokens = settings.gemini_max_tokens
        logger.info("Gemini client initialized (model=%s)", self._model)

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

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        text = response.text
        if text is None:
            raise RuntimeError("Gemini returned empty response")

        logger.debug(
            "Gemini response: %d chars, finish=%s",
            len(text),
            response.candidates[0].finish_reason if response.candidates else "N/A",
        )
        return text

    def generate_with_image(
        self,
        text_prompt: str,
        image_bytes: bytes,
        *,
        system_instruction: str | None = None,
        **kwargs,
    ) -> str:
        """Convenience: generate with a text prompt + screenshot image."""
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        return self.generate(
            contents=[text_prompt, image_part],
            system_instruction=system_instruction,
            **kwargs,
        )
