"""Named Pipe control server – receives commands from external PowerShell clients."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from queue import Queue

import win32file
import win32pipe

from itacolumite.config.settings import get_settings

logger = logging.getLogger(__name__)

PIPE_BUFFER_SIZE = 4096


class ControlCommand(Enum):
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    SEND = "send"


@dataclass
class ControlMessage:
    """A parsed control command from the pipe."""

    command: ControlCommand
    payload: str = ""


def _parse_message(raw: str) -> ControlMessage | None:
    """Parse raw pipe data into a ControlMessage."""
    raw = raw.strip()
    if not raw:
        return None

    if raw.startswith("send:"):
        return ControlMessage(command=ControlCommand.SEND, payload=raw[5:].strip())

    try:
        cmd = ControlCommand(raw.lower())
        return ControlMessage(command=cmd)
    except ValueError:
        logger.warning("Unknown control command: %s", raw)
        return None


class ControlServer:
    """Named Pipe server that runs in a background thread.

    Receives control commands and puts them on a queue
    for the agent loop to consume.
    """

    def __init__(self, command_queue: Queue[ControlMessage] | None = None) -> None:
        settings = get_settings()
        self._pipe_name = rf"\\.\pipe\{settings.agent.control_pipe_name}"
        self._queue: Queue[ControlMessage] = command_queue or Queue()
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def queue(self) -> Queue[ControlMessage]:
        return self._queue

    def start(self) -> None:
        """Start the pipe server in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True, name="control-pipe")
        self._thread.start()
        logger.info("Control server started on %s", self._pipe_name)

    def stop(self) -> None:
        """Signal the server to stop."""
        self._running = False
        # Connect to the pipe to unblock WaitNamedPipe / ConnectNamedPipe
        try:
            handle = win32file.CreateFile(
                self._pipe_name,
                win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None,
            )
            win32file.WriteFile(handle, b"")
            win32file.CloseHandle(handle)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        logger.info("Control server stopped")

    def _serve_loop(self) -> None:
        """Main server loop — create pipe, wait for connection, read, repeat."""
        while self._running:
            try:
                pipe_handle = win32pipe.CreateNamedPipe(
                    self._pipe_name,
                    win32pipe.PIPE_ACCESS_INBOUND,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                    1,  # max instances
                    PIPE_BUFFER_SIZE,
                    PIPE_BUFFER_SIZE,
                    0,  # default timeout
                    None,  # default security
                )

                # Block until a client connects
                win32pipe.ConnectNamedPipe(pipe_handle, None)

                # Read the message
                try:
                    _, data = win32file.ReadFile(pipe_handle, PIPE_BUFFER_SIZE)
                    if data:
                        text = data.decode("utf-8", errors="replace")
                        msg = _parse_message(text)
                        if msg is not None:
                            self._queue.put(msg)
                            logger.info("Control command received: %s", msg.command.value)
                except Exception as e:
                    if self._running:
                        logger.debug("Pipe read error (expected on shutdown): %s", e)
                finally:
                    win32file.CloseHandle(pipe_handle)

            except Exception as e:
                if self._running:
                    logger.error("Control server error: %s", e)
