"""Copilot Chat interaction – the ONLY path for source code modification."""

from __future__ import annotations

import logging
import time

from itacolumite.action.keyboard import KeyboardController
from itacolumite.action.mouse import MouseController

logger = logging.getLogger(__name__)


class CopilotTask:
    """Manages interaction with GitHub Copilot Chat inside VS Code.

    This is the ONLY sanctioned way for the agent to modify source code.
    The workflow:
      1. Open Copilot Chat
      2. Write a prompt describing the desired change
      3. Wait for Copilot's response
      4. Accept/reject the suggestion
      5. Save the file
    """

    def __init__(
        self,
        keyboard: KeyboardController,
        mouse: MouseController,
    ) -> None:
        self._keyboard = keyboard
        self._mouse = mouse

    def open_chat(self) -> None:
        """Open the Copilot Chat panel via keyboard shortcut."""
        # Ctrl+Shift+I opens Copilot Chat in VS Code
        self._keyboard.combo("ctrl-shift-i")
        time.sleep(1.0)
        logger.info("Copilot Chat opened")

    def close_chat(self) -> None:
        """Close the Copilot Chat panel."""
        self._keyboard.combo("ctrl-shift-i")
        time.sleep(0.5)

    def send_prompt(self, prompt: str) -> None:
        """Type a prompt into Copilot Chat and send it.

        The Chat panel should already be open and focused.
        """
        # Type the prompt
        self._keyboard.type_text(prompt)
        time.sleep(0.3)

        # Send with Enter
        self._keyboard.enter()
        logger.info("Copilot prompt sent: %s", prompt[:80] + "..." if len(prompt) > 80 else prompt)

    def wait_for_response(self, timeout: float = 30.0) -> None:
        """Wait for Copilot to finish generating a response.

        In practice, the agent's main loop will capture a screenshot
        after this delay and use Gemini to determine if Copilot is done.
        """
        time.sleep(timeout)

    def open_inline_chat(self) -> None:
        """Open inline Copilot Chat (Ctrl+I in editor)."""
        self._keyboard.combo("ctrl-i")
        time.sleep(0.5)

    def accept_suggestion(self) -> None:
        """Accept the current Copilot suggestion."""
        self._keyboard.tab()
        time.sleep(0.3)

    def reject_suggestion(self) -> None:
        """Reject the current Copilot suggestion."""
        self._keyboard.escape()
        time.sleep(0.3)
