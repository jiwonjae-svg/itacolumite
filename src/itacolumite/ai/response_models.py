"""Response models for parsing Gemini structured output."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class ActionParams(BaseModel):
    """Parameters for an agent action."""

    model_config = ConfigDict(extra="allow")

    # Mouse actions
    x: int | None = None
    y: int | None = None
    x1: int | None = None
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None
    button: str | None = None
    direction: str | None = None
    amount: int | None = None

    # Keyboard/text actions
    text: str | None = None
    key: str | None = None
    keys: str | None = None

    # Shell actions (structured)
    program: str | None = None
    args: list[str] | None = None
    cwd: str | None = None
    timeout: int | None = None

    # Wait action
    seconds: float | None = None

    # Task complete
    result: str | None = None


class AgentAction(BaseModel):
    """A single action the agent wants to perform."""

    type: str
    params: ActionParams = Field(default_factory=ActionParams)


class AgentResponse(BaseModel):
    """Parsed response from Gemini."""

    observation: str = ""
    reasoning: str = ""
    plan: list[str] = Field(default_factory=list)
    next_action: AgentAction
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def parse_agent_response(raw: str) -> AgentResponse:
    """Parse raw JSON string from Gemini into an AgentResponse.

    Handles common issues like markdown code fences.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON: %s\nRaw: %s", e, text[:500])
        raise ValueError(f"Invalid JSON from Gemini: {e}") from e

    return AgentResponse.model_validate(data)
