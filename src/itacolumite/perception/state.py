"""System state collection for the perception layer (native Windows)."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field

from itacolumite.perception.window import get_foreground_window

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Snapshot of the local system state."""

    cwd: str = ""
    foreground_window: str = ""
    processes: str = ""
    git_status: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


class StateCollector:
    """로컬 Windows 시스템 상태 수집 (read-only)."""

    def collect(self, task_type: str | None = None) -> SystemState:
        state = SystemState()
        state.cwd = os.getcwd()

        try:
            winfo = get_foreground_window()
            state.foreground_window = f"{winfo.title} [{winfo.class_name}]"
        except Exception as e:
            state.foreground_window = f"[error: {e}]"

        state.processes = self._safe_ps("Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Name, CPU, Id | Format-Table -AutoSize")

        if task_type == "coding":
            state.git_status = self._safe_ps("git status --short 2>$null")

        return state

    def _safe_ps(self, cmd: str) -> str:
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", cmd],
                shell=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception as e:
            logger.error("PowerShell command failed: %s", e)
            return f"[error: {e}]"
