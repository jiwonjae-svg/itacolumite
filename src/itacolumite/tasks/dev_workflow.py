"""Full development workflow – the high-level orchestration for coding tasks."""

from __future__ import annotations

import logging

from itacolumite.tasks.copilot import CopilotTask
from itacolumite.tasks.terminal import TerminalTask
from itacolumite.tasks.vscode import VSCodeTask

logger = logging.getLogger(__name__)


class DevWorkflow:
    """Orchestrates the full development cycle:

    1. Open VS Code → 2. Open Copilot Chat → 3. Send prompt →
    4. Wait for response → 5. Save → 6. Build/Test → 7. Iterate
    """

    def __init__(
        self,
        vscode: VSCodeTask,
        copilot: CopilotTask,
        terminal: TerminalTask,
    ) -> None:
        self._vscode = vscode
        self._copilot = copilot
        self._terminal = terminal

    def coding_cycle(self, prompt: str, test_program: str | None = None, test_args: list[str] | None = None) -> None:
        """Execute one code-generate-test cycle.

        1. Open Copilot Chat
        2. Send the prompt
        3. Wait for response
        4. Save all files
        5. Run tests if program provided
        """
        self._copilot.open_chat()
        self._copilot.send_prompt(prompt)
        self._copilot.wait_for_response(timeout=15.0)
        self._vscode.save_all()

        if test_program:
            try:
                output = self._terminal.run_tests(program=test_program, args=test_args)
                logger.info("Tests output:\n%s", output[:500])
            except RuntimeError as e:
                logger.warning("Tests failed: %s", e)

    def setup_project(self) -> None:
        """Check git status in the workspace."""
        self._terminal.git_status()
        logger.info("Project git status checked")
