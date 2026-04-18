"""VS Code operation helpers."""

from __future__ import annotations

import logging
import time

from itacolumite.action.keyboard import KeyboardController
from itacolumite.action.mouse import MouseController

logger = logging.getLogger(__name__)


class VSCodeTask:
    """High-level VS Code operations."""

    def __init__(
        self,
        keyboard: KeyboardController,
        mouse: MouseController,
    ) -> None:
        self._keyboard = keyboard
        self._mouse = mouse

    def launch(self, path: str = ".") -> None:
        """Launch VS Code."""
        import subprocess
        subprocess.Popen(["code", path], shell=False)
        time.sleep(3)
        logger.info("VS Code launched at %s", path)

    def open_file(self, filepath: str) -> None:
        """Open a file via Ctrl+P quick open."""
        self._keyboard.combo("ctrl-p")
        time.sleep(0.5)
        self._keyboard.type_text(filepath)
        time.sleep(0.3)
        self._keyboard.enter()
        time.sleep(0.5)

    def open_terminal(self) -> None:
        """Open VS Code integrated terminal."""
        self._keyboard.combo("ctrl-grave")
        time.sleep(0.5)

    def open_command_palette(self) -> None:
        """Open command palette."""
        self._keyboard.combo("ctrl-shift-p")
        time.sleep(0.5)

    def save_file(self) -> None:
        """Save the current file."""
        self._keyboard.save()
        time.sleep(0.3)

    def save_all(self) -> None:
        """Save all open files."""
        self._keyboard.combo("ctrl-shift-s")
        time.sleep(0.3)

    def close_file(self) -> None:
        """Close current editor tab."""
        self._keyboard.combo("ctrl-w")
        time.sleep(0.3)

    def new_file(self) -> None:
        """Create a new untitled file."""
        self._keyboard.combo("ctrl-n")
        time.sleep(0.3)

    def go_to_line(self, line: int) -> None:
        """Jump to a specific line number."""
        self._keyboard.combo("ctrl-g")
        time.sleep(0.3)
        self._keyboard.type_text(str(line))
        self._keyboard.enter()

    def find_in_file(self, query: str) -> None:
        """Open find dialog and search."""
        self._keyboard.combo("ctrl-f")
        time.sleep(0.3)
        self._keyboard.type_text(query)

    def find_and_replace(self, find: str, replace: str) -> None:
        """Open find-and-replace dialog."""
        self._keyboard.combo("ctrl-h")
        time.sleep(0.3)
        self._keyboard.type_text(find)
        self._keyboard.tab()
        self._keyboard.type_text(replace)
