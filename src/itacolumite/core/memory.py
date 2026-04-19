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
        # Archive of condensed-away actions (preserved for persistence)
        self._full_history_archive: list[ActionRecord] = []
        self._step_counter = 0

        # Current task context
        self._current_task: str = ""
        self._task_id: str = ""
        self._task_start: str = ""

        # History summarization: condensed text of old steps
        self._condensed_summary: str = ""
        self._condense_threshold: int = max_short_term

    # ── Short-term memory ────────────────────────────────────

    def record_action(self, record: ActionRecord) -> None:
        """Add an action to short-term and full history."""
        self._short_term.append(record)
        self._full_history.append(record)
        logger.debug("Recorded action step %d: %s", record.step, record.action_type)

        # Auto-condense when history exceeds 2× the short-term window
        if len(self._full_history) > self._condense_threshold * 2:
            self._condense_old_history()

    def next_step(self) -> int:
        """Get and increment the step counter."""
        self._step_counter += 1
        return self._step_counter

    def get_recent_history(self, n: int = 10) -> list[ActionRecord]:
        """Get the N most recent actions."""
        items = list(self._short_term)
        return items[-n:]

    def get_history_summary(self, n: int = 10) -> str:
        """Get a text summary of recent actions for prompt context.

        Includes a condensed summary of older steps if available.
        """
        recent = self.get_recent_history(n)
        if not recent and not self._condensed_summary:
            return "No actions taken yet."
        parts: list[str] = []
        if self._condensed_summary:
            parts.append(f"[Earlier actions summary]\n{self._condensed_summary}\n")
        if recent:
            parts.append("\n".join(r.summary() for r in recent))
        return "\n".join(parts)

    @property
    def step_count(self) -> int:
        return self._step_counter

    def _condense_old_history(self) -> None:
        """Compress older history entries into a short text summary.

        Keeps the most recent `_condense_threshold` actions in full
        and summarises everything older into `_condensed_summary`.
        """
        keep_count = self._condense_threshold
        if len(self._full_history) <= keep_count:
            return
        old = self._full_history[:-keep_count]
        # Build a compact summary: aggregate action types + success/failure counts
        type_counts: dict[str, dict[str, int]] = {}
        for rec in old:
            bucket = type_counts.setdefault(rec.action_type, {"success": 0, "failure": 0})
            if rec.result == "success":
                bucket["success"] += 1
            else:
                bucket["failure"] += 1
        lines = [f"Steps 1-{old[-1].step} condensed ({len(old)} actions):"]
        for atype, counts in type_counts.items():
            lines.append(f"  {atype}: {counts['success']} ok, {counts['failure']} fail")
        self._condensed_summary = "\n".join(lines)
        # Trim full_history to keep only recent (keep archive for persistence)
        if not hasattr(self, '_full_history_archive'):
            self._full_history_archive: list[ActionRecord] = []
        self._full_history_archive.extend(old)
        self._full_history = self._full_history[-keep_count:]
        logger.info("History condensed: %d old actions summarised", len(old))

    # ── Task context ─────────────────────────────────────────

    def start_task(self, task_id: str, description: str) -> None:
        """Begin tracking a new task."""
        self._task_id = task_id
        self._current_task = description
        self._task_start = datetime.now().isoformat()
        self._step_counter = 0
        self._short_term.clear()
        self._full_history.clear()
        self._full_history_archive.clear()
        self._condensed_summary = ""
        logger.info("Task started: %s – %s", task_id, description)

    def end_task(self, result: str, *, token_usage: dict[str, int] | None = None) -> None:
        """End the current task and save to long-term storage."""
        if not self._task_id:
            return

        task_record: dict[str, Any] = {
            "task_id": self._task_id,
            "description": self._current_task,
            "started": self._task_start,
            "ended": datetime.now().isoformat(),
            "total_steps": self._step_counter,
            "result": result,
            "actions": [asdict(r) for r in self._full_history_archive + self._full_history],
        }
        if token_usage:
            task_record["token_usage"] = token_usage

        # Save to long-term storage
        task_file = self._data_dir / "task_history" / f"{self._task_id}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(json.dumps(task_record, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Task ended: %s – %s (%d steps)", self._task_id, result, self._step_counter)

        self._task_id = ""
        self._current_task = ""

    @property
    def current_task(self) -> str:
        return self._current_task

    @property
    def task_id(self) -> str:
        return self._task_id

    # ── Checkpoint (state persistence) ───────────────────────

    def save_checkpoint(self, state_dict: dict[str, Any]) -> None:
        """Persist a step-level checkpoint so the task can be resumed."""
        if not self._task_id:
            return
        cp_file = self._data_dir / "checkpoints" / f"{self._task_id}.json"
        cp_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": self._task_id,
            "description": self._current_task,
            "started": self._task_start,
            "step": self._step_counter,
            "state": state_dict,
            "actions": [asdict(r) for r in self._full_history_archive + self._full_history],
        }
        cp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, task_id: str) -> dict[str, Any] | None:
        """Load a previously saved checkpoint, or return None."""
        settings = get_settings()
        cp_file = settings.agent_data_dir / "memory" / "checkpoints" / f"{task_id}.json"
        if not cp_file.exists():
            return None
        return json.loads(cp_file.read_text(encoding="utf-8"))

    @classmethod
    def latest_checkpoint_id(cls) -> str | None:
        """Return the task_id of the most recent checkpoint file, or None."""
        settings = get_settings()
        cp_dir = settings.agent_data_dir / "memory" / "checkpoints"
        if not cp_dir.exists():
            return None
        files = sorted(cp_dir.glob("task-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        data = json.loads(files[0].read_text(encoding="utf-8"))
        return data.get("task_id")
