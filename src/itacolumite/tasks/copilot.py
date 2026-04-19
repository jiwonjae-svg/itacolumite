"""Copilot Chat interaction – the ONLY path for source code modification."""

from __future__ import annotations

import logging
import time

from itacolumite.action.keyboard import KeyboardController
from itacolumite.action.mouse import MouseController
from itacolumite.perception.screen import ScreenCapture

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 1.0  # seconds between screenshots
_STABLE_ROUNDS = 3    # consecutive unchanged frames to declare "done"
_DIFF_THRESHOLD = 0.001  # fraction of changed pixels to count as "changed"


def _frames_similar(a: bytes, b: bytes) -> bool:
    """Return True if two PNG byte-buffers represent nearly identical images."""
    if len(a) != len(b):
        return False
    if a == b:
        return True
    # Byte-level comparison: count differing bytes as a rough pixel-diff proxy.
    diff_count = sum(1 for x, y in zip(a, b) if x != y)
    return diff_count / max(len(a), 1) < _DIFF_THRESHOLD


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
        screen: ScreenCapture | None = None,
    ) -> None:
        self._keyboard = keyboard
        self._mouse = mouse
        self._screen = screen

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

    def wait_for_response(self, timeout: float = 30.0) -> bool:
        """Wait for Copilot to finish generating a response.

        Uses screenshot polling when a ScreenCapture instance is available:
        captures frames every _POLL_INTERVAL seconds and considers the response
        complete once _STABLE_ROUNDS consecutive frames are nearly identical.

        Returns True if stability was detected, False if the timeout was hit.
        Falls back to a fixed sleep when no screen capture is available.
        """
        if self._screen is None:
            time.sleep(timeout)
            return False

        deadline = time.monotonic() + timeout
        prev_bytes: bytes | None = None
        stable_count = 0

        while time.monotonic() < deadline:
            frame = self._screen.capture_bytes()
            if prev_bytes is not None and _frames_similar(prev_bytes, frame):
                stable_count += 1
                if stable_count >= _STABLE_ROUNDS:
                    logger.info("Copilot response stabilised after %d rounds", stable_count)
                    return True
            else:
                stable_count = 0
            prev_bytes = frame
            remaining = deadline - time.monotonic()
            time.sleep(min(_POLL_INTERVAL, max(remaining, 0)))

        logger.warning("Copilot response wait timed out after %.1fs", timeout)
        return False

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
