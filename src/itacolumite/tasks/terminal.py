"""Terminal operations (PowerShell based)."""

from __future__ import annotations

import logging

from itacolumite.action.keyboard import KeyboardController
from itacolumite.action.shell import ShellController, ShellRequest

logger = logging.getLogger(__name__)


class TerminalTask:
    """Terminal operations – build, test, and status checking."""

    def __init__(
        self,
        keyboard: KeyboardController,
        shell: ShellController,
    ) -> None:
        self._keyboard = keyboard
        self._shell = shell

    def run_build(self, program: str = "npm", args: list[str] | None = None) -> str:
        args = args or ["run", "build"]
        result = self._shell.execute(ShellRequest(program=program, args=args))
        if not result.success:
            detail = result.output or result.error
            raise RuntimeError(f"Build failed: {detail}")
        return result.output

    def run_tests(self, program: str = "pytest", args: list[str] | None = None) -> str:
        args = args or []
        result = self._shell.execute(ShellRequest(program=program, args=args))
        if not result.success:
            detail = result.output or result.error
            raise RuntimeError(f"Tests failed: {detail}")
        return result.output

    def install_packages(self, program: str = "pip", args: list[str] | None = None) -> str:
        args = args or ["install", "-r", "requirements.txt"]
        result = self._shell.execute(ShellRequest(program=program, args=args))
        if not result.success:
            detail = result.output or result.error
            raise RuntimeError(f"Install failed: {detail}")
        return result.output

    def git_status(self) -> str:
        result = self._shell.execute(ShellRequest(program="git", args=["status"]))
        return result.output

    def git_commit(self, message: str) -> str:
        self._shell.execute(ShellRequest(program="git", args=["add", "-A"]), force=True)
        result = self._shell.execute(ShellRequest(program="git", args=["commit", "-m", message]), force=True)
        return result.output
