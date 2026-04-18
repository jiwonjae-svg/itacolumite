"""Memory system – short-term and long-term memory for the agent."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from itacolumite.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """Record of a single action taken by the agent."""

    step: int
    timestamp: str
    action_type: str
    params: dict[str, Any]
    observation: str
    reasoning: str
    confidence: float
    result: str = ""  # "success", "failure", "blocked"
    verification: str = ""

    def summary(self) -> str:
        """One-line summary for prompt context."""
        p = json.dumps(self.params, ensure_ascii=False) if self.params else "{}"
        return f"[Step {self.step}] {self.action_type}({p}) → {self.result}"


class Memory:
    """Agent memory system with short-term history and persistent long-term storage."""

    def __init__(self, max_short_term: int = 20) -> None:
        self._settings = get_settings()
        self._data_dir = self._settings.agent_data_dir / "memory"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Short-term: recent N actions (used for prompt context)
        self._short_term: deque[ActionRecord] = deque(maxlen=max_short_term)
        # Full history: all actions for persistence
        self._full_history: list[ActionRecord] = []
        self._step_counter = 0

        # Current task context
        self._current_task: str = ""
        self._task_id: str = ""
        self._task_start: str = ""

    # ── Short-term memory ────────────────────────────────────

    def record_action(self, record: ActionRecord) -> None:
        """Add an action to short-term and full history."""
        self._short_term.append(record)
        self._full_history.append(record)
        logger.debug("Recorded action step %d: %s", record.step, record.action_type)

    def next_step(self) -> int:
        """Get and increment the step counter."""
        self._step_counter += 1
        return self._step_counter

    def get_recent_history(self, n: int = 10) -> list[ActionRecord]:
        """Get the N most recent actions."""
        items = list(self._short_term)
        return items[-n:]

    def get_history_summary(self, n: int = 10) -> str:
        """Get a text summary of recent actions for prompt context."""
        recent = self.get_recent_history(n)
        if not recent:
            return "No actions taken yet."
        return "\n".join(r.summary() for r in recent)

    @property
    def step_count(self) -> int:
        return self._step_counter

    # ── Task context ─────────────────────────────────────────

    def start_task(self, task_id: str, description: str) -> None:
        """Begin tracking a new task."""
        self._task_id = task_id
        self._current_task = description
        self._task_start = datetime.now().isoformat()
        self._step_counter = 0
        self._short_term.clear()
        self._full_history.clear()
        logger.info("Task started: %s – %s", task_id, description)

    def end_task(self, result: str) -> None:
        """End the current task and save to long-term storage."""
        if not self._task_id:
            return

        task_record = {
            "task_id": self._task_id,
            "description": self._current_task,
            "started": self._task_start,
            "ended": datetime.now().isoformat(),
            "total_steps": self._step_counter,
            "result": result,
            "actions": [asdict(r) for r in self._full_history],
        }

        # Save to long-term storage
        task_file = self._data_dir / "task_history" / f"{self._task_id}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(json.dumps(task_record, ensure_ascii=False, indent=2))
        logger.info("Task ended: %s – %s (%d steps)", self._task_id, result, self._step_counter)

        self._task_id = ""
        self._current_task = ""

    @property
    def current_task(self) -> str:
        return self._current_task

    @property
    def task_id(self) -> str:
        return self._task_id
