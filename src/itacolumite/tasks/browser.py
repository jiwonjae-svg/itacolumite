"""Browser operations on native Windows."""

from __future__ import annotations

import logging
import subprocess
import time

from itacolumite.action.keyboard import KeyboardController
from itacolumite.action.mouse import MouseController

logger = logging.getLogger(__name__)


class BrowserTask:
    """Chrome browser operations."""

    def __init__(
        self,
        keyboard: KeyboardController,
        mouse: MouseController,
    ) -> None:
        self._keyboard = keyboard
        self._mouse = mouse

    def launch(self, url: str = "") -> None:
        """Launch Chrome."""
        cmd = ["chrome"]
        if url:
            cmd.append(url)
        subprocess.Popen(cmd, shell=False)
        time.sleep(3)
        logger.info("Chrome launched")

    def navigate(self, url: str) -> None:
        """Navigate to a URL via address bar."""
        self._keyboard.combo("ctrl-l")
        time.sleep(0.3)
        self._keyboard.select_all()
        self._keyboard.type_text(url)
        self._keyboard.enter()
        time.sleep(2)

    def new_tab(self) -> None:
        self._keyboard.combo("ctrl-t")
        time.sleep(0.5)

    def close_tab(self) -> None:
        self._keyboard.combo("ctrl-w")
        time.sleep(0.3)

    def go_back(self) -> None:
        self._keyboard.combo("alt-Left")
        time.sleep(0.5)

    def go_forward(self) -> None:
        self._keyboard.combo("alt-Right")
        time.sleep(0.5)

    def refresh(self) -> None:
        self._keyboard.combo("ctrl-r")
        time.sleep(1)

    def search(self, query: str) -> None:
        """Search via address bar."""
        self.navigate(f"https://www.google.com/search?q={query}")
