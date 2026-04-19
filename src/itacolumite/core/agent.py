"""Main agent loop – Observe → Analyze → Plan → Act → Verify."""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty, Queue
from typing import Any

from itacolumite.action.clipboard import ClipboardController
from itacolumite.action.keyboard import KeyboardController
from itacolumite.action.mouse import MouseController
from itacolumite.action.shell import ShellController
from itacolumite.ai.gemini_client import GeminiClient
from itacolumite.ai.prompts.system import SYSTEM_PROMPT, build_observe_prompt
from itacolumite.ai.response_models import AgentResponse, parse_agent_response
from itacolumite.config.settings import get_settings
from itacolumite.core.coordinate_validation import (
    ValidationConfig,
    needs_coordinate_validation,
    validate_action_coordinates,
)
from itacolumite.core.executor import ActionExecutor, ExecutionResult
from itacolumite.core.grounding_capture import (
    GeminiGroundingExtractor,
    save_grounding_capture_image,
    write_grounding_provider_payload,
)
from itacolumite.core.omniparser_runner import OmniParserRunner
from itacolumite.core.grounding_providers import build_default_grounding_providers
from itacolumite.core.grounding_telemetry import GroundingTelemetryLogger
from itacolumite.core.memory import ActionRecord, Memory
from itacolumite.interface.control_server import ControlCommand, ControlMessage, ControlServer
from itacolumite.perception.screen import CaptureContext, ScreenCapture
from itacolumite.perception.state import StateCollector

logger = logging.getLogger(__name__)

# ── Error recovery constants ─────────────────────────────────

_MAX_CONSECUTIVE_FAILURES = 5
_MAX_IDENTICAL_ACTIONS = 4
_LOOP_HISTORY_SIZE = 10


@dataclass
class StepSnapshot:
    """Lightweight snapshot of a step for loop detection."""

    action_type: str
    params_hash: int
    observation_prefix: str


@dataclass
class AgentState:
    """Observable state of the agent for Rich display."""

    task: str = ""
    task_id: str = ""
    step: int = 0
    max_steps: int = 200
    running: bool = False
    paused: bool = False
    observation: str = ""
    reasoning: str = ""
    plan: list[str] = field(default_factory=list)
    next_action: str = ""
    confidence: float = 1.0
    current_model: str = ""
    last_result: str = ""
    screenshot_path: str = ""
    api_calls: int = 0
    actions_taken: int = 0
    consecutive_failures: int = 0
    start_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Agent:
    """The core autonomous agent.

    Implements the Observe → Analyze → Plan → Act → Verify loop
    with control channel integration and error recovery.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

        # Perception
        self._screen = ScreenCapture()
        self._state_collector = StateCollector()

        # Action
        mouse = MouseController()
        keyboard = KeyboardController()
        shell = ShellController()
        clipboard = ClipboardController()
        self._executor = ActionExecutor(mouse, keyboard, shell, clipboard)

        # AI
        self._gemini = GeminiClient()

        # Memory
        self._memory = Memory()

        # Control channel
        self._control_server = ControlServer()
        self._control_queue: Queue[ControlMessage] = self._control_server.queue

        # State
        self._running = False
        self._paused = False
        self._user_messages: list[str] = []
        self._agent_state = AgentState()

        # Error recovery
        self._consecutive_failures = 0
        self._recent_actions: deque[StepSnapshot] = deque(maxlen=_LOOP_HISTORY_SIZE)
        self._grounding_config = ValidationConfig(
            require_bbox=self._settings.grounding.grounding_require_bbox,
            min_confidence=self._settings.grounding.grounding_min_confidence,
            edge_margin_px=self._settings.grounding.grounding_edge_margin_px,
            min_bbox_size_px=self._settings.grounding.grounding_min_bbox_size_px,
            max_repeat_failures=self._settings.grounding.grounding_max_repeat_failures,
        )
        self._grounding_providers = build_default_grounding_providers(self._settings)
        self._grounding_telemetry = GroundingTelemetryLogger(self._settings.agent_data_dir)
        self._grounding_extractor = (
            GeminiGroundingExtractor(self._gemini)
            if self._settings.grounding.grounding_enable_gemini_ocr_provider
            else None
        )
        self._omniparser_runner = (
            OmniParserRunner.from_settings(self._settings)
            if self._settings.grounding.grounding_enable_omniparser_runner
            else None
        )

    @property
    def state(self) -> AgentState:
        """Read-only snapshot of the current agent state."""
        return self._agent_state

    # ── Public API ───────────────────────────────────────────

    def start(self) -> None:
        """에이전트 시작."""
        self._running = True
        self._paused = False
        self._control_server.start()
        logger.info("Agent started.")

    def stop(self) -> None:
        """에이전트 정지."""
        self._running = False
        self._control_server.stop()
        logger.info("Agent stopped.")

    def run_task(self, task_description: str, *, resume_task_id: str | None = None) -> str:
        """Run a task to completion.

        If *resume_task_id* is given, restore the checkpoint and continue from
        where the previous run left off.

        Returns the task result string.
        """
        start_step = 0

        if resume_task_id:
            cp = Memory.load_checkpoint(resume_task_id)
            if cp is None:
                logger.error("No checkpoint found for %s", resume_task_id)
                return "error: checkpoint_not_found"
            task_id = resume_task_id
            task_description = cp.get("description", task_description)
            start_step = cp.get("step", 0)
            self._memory.start_task(task_id, task_description)
            # Restore history from checkpoint
            for rec_dict in cp.get("actions", []):
                self._memory.record_action(ActionRecord(**rec_dict))
            self._memory._step_counter = start_step
            logger.info("Resuming task %s from step %d", task_id, start_step)
        else:
            task_id = f"task-{uuid.uuid4().hex[:8]}"
            self._memory.start_task(task_id, task_description)

        self._consecutive_failures = 0
        self._recent_actions.clear()
        self._user_messages.clear()

        # Update observable state
        s = self._agent_state
        s.task = task_description
        s.task_id = task_id
        s.max_steps = self._settings.agent.agent_max_steps
        s.running = True
        s.start_time = time.time()

        logger.info("═" * 60)
        logger.info("Task: %s", task_description)
        logger.info("═" * 60)

        max_steps = self._settings.agent.agent_max_steps
        remaining_steps = max_steps - start_step
        result = "max_steps_reached"

        try:
            for _ in range(remaining_steps):
                # Process control commands
                self._process_control_commands()

                if not self._running:
                    result = "agent_stopped"
                    break

                # Handle pause
                while self._paused and self._running:
                    time.sleep(0.5)
                    self._process_control_commands()

                if not self._running:
                    result = "agent_stopped"
                    break

                step_result = self._execute_step(task_description)

                if step_result.task_complete:
                    result = step_result.task_result
                    break

                # Error recovery: too many consecutive failures
                if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        "Too many consecutive failures (%d). Injecting recovery hint.",
                        self._consecutive_failures,
                    )
                    self._user_messages.append(
                        "WARNING: You have failed multiple times in a row. "
                        "Try a completely different approach or simplify your plan."
                    )
                    self._consecutive_failures = 0

        except KeyboardInterrupt:
            result = "interrupted_by_user"
            logger.info("Task interrupted by user.")
        except Exception as e:
            result = f"error: {e}"
            logger.exception("Task failed with error:")
        finally:
            usage = self._gemini.usage
            self._memory.end_task(
                result,
                token_usage={
                    "total_calls": usage.total_calls,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
            )
            s.running = False

        logger.info("═" * 60)
        logger.info("Task result: %s", result)
        logger.info("═" * 60)

        return result

    # ── Control channel ──────────────────────────────────────

    def _process_control_commands(self) -> None:
        """Drain the control queue and handle each command."""
        while True:
            try:
                msg = self._control_queue.get_nowait()
            except Empty:
                break

            if msg.command == ControlCommand.PAUSE:
                self._paused = True
                self._agent_state.paused = True
                logger.info("Agent paused by control command.")
            elif msg.command == ControlCommand.RESUME:
                self._paused = False
                self._agent_state.paused = False
                logger.info("Agent resumed by control command.")
            elif msg.command == ControlCommand.STOP:
                self._running = False
                logger.info("Agent stopped by control command.")
            elif msg.command == ControlCommand.SEND:
                self._user_messages.append(msg.payload)
                logger.info("User message received: %s", msg.payload[:80])

    # ── Agent loop step ──────────────────────────────────────

    def _execute_step(self, task: str) -> ExecutionResult:
        """Execute one observe→analyze→act→verify cycle."""
        step = self._memory.next_step()
        s = self._agent_state
        s.step = step
        logger.info("─── Step %d ───", step)

        # 1. OBSERVE: Capture screen + internal state
        screenshot_bytes, capture_context = self._screen.capture_bytes_with_context()
        internal_state = self._state_collector.collect(task_type=self._infer_task_type(task))
        state_text = self._format_state(internal_state)

        # Update focus guard: inform executor of the current foreground window
        try:
            from itacolumite.perception.window import get_foreground_window
            winfo = get_foreground_window()
            if winfo and winfo.title:
                self._executor.set_expected_window(winfo)
        except Exception:
            pass  # non-critical — guard stays disabled

        # Inject user messages into context
        user_context = ""
        if self._user_messages:
            user_context = "\n\n## User Messages\n" + "\n".join(
                f"- {m}" for m in self._user_messages
            )
            self._user_messages.clear()

        # 2. ANALYZE + PLAN: Send to Gemini
        history_summary = self._memory.get_history_summary()
        screenshot_context = self._format_capture_context(capture_context)
        user_prompt = build_observe_prompt(
            task,
            history_summary,
            state_text + user_context,
            screenshot_context,
        )

        gemini_cfg = self._settings.gemini
        use_pro = gemini_cfg.gemini_use_pro_for_vision
        if not use_pro and gemini_cfg.gemini_auto_upgrade:
            use_pro = (
                self._consecutive_failures >= 2
                or (step > 1 and s.confidence < gemini_cfg.gemini_auto_upgrade_threshold)
            )
        s.current_model = (
            gemini_cfg.gemini_model_pro if use_pro else gemini_cfg.gemini_model_fast
        )
        if use_pro:
            logger.info(
                "[Auto-upgrade] Pro model selected (failures=%d, confidence=%.2f)",
                self._consecutive_failures,
                s.confidence,
            )

        s.api_calls += 1
        raw_response = self._gemini.generate_with_image(
            text_prompt=user_prompt,
            image_bytes=screenshot_bytes,
            system_instruction=SYSTEM_PROMPT,
            use_pro=use_pro,
        )
        # Update token usage in observable state
        usage = self._gemini.usage
        s.prompt_tokens = usage.prompt_tokens
        s.completion_tokens = usage.completion_tokens
        s.total_tokens = usage.total_tokens

        # 3. Parse response
        try:
            response = parse_agent_response(raw_response)
        except ValueError as e:
            logger.error("Failed to parse response: %s", e)
            self._consecutive_failures += 1
            record = ActionRecord(
                step=step,
                timestamp=datetime.now().isoformat(),
                action_type="parse_error",
                params={},
                observation="",
                reasoning=str(e),
                confidence=0.0,
                result="failure",
            )
            self._memory.record_action(record)
            return ExecutionResult(success=False, action_type="parse_error", error=str(e))

        # Update observable state
        s.observation = response.observation
        s.reasoning = response.reasoning
        s.plan = response.plan
        s.next_action = f"{response.next_action.type}({response.next_action.params.model_dump(exclude_none=True)})"
        s.confidence = response.confidence

        logger.info(
            "Observation: %s",
            response.observation[:100] + "..." if len(response.observation) > 100 else response.observation,
        )
        logger.info("Reasoning: %s", response.reasoning[:100] + "..." if len(response.reasoning) > 100 else response.reasoning)
        logger.info("Action: %s (confidence=%.2f)", response.next_action.type, response.confidence)

        validation = None
        pre_action_img = self._screen.last_screenshot
        if needs_coordinate_validation(response.next_action.type):
            self._maybe_refresh_grounding_providers(screenshot_bytes, capture_context)
            validation = validate_action_coordinates(
                response.next_action,
                confidence=response.confidence,
                capture_context=capture_context,
                recent_records=self._memory.get_recent_history(),
                config=self._grounding_config,
                screenshot=pre_action_img,
                task=task,
                providers=self._grounding_providers,
            )
            self._grounding_telemetry.record_validation(
                task_id=s.task_id,
                step=step,
                model=s.current_model,
                confidence=response.confidence,
                action=response.next_action,
                validation=validation,
            )
            if response.next_action.type == "mouse_drag":
                if validation.pixel_point is not None:
                    response.next_action.params.x1 = validation.pixel_point[0]
                    response.next_action.params.y1 = validation.pixel_point[1]
                if validation.pixel_point_end is not None:
                    response.next_action.params.x2 = validation.pixel_point_end[0]
                    response.next_action.params.y2 = validation.pixel_point_end[1]
            elif validation.pixel_point is not None:
                response.next_action.params.x = validation.pixel_point[0]
                response.next_action.params.y = validation.pixel_point[1]
            if validation.retry_hint:
                response.next_action.params.validation_note = validation.retry_hint
            if not validation.approved:
                logger.warning(
                    "Coordinate validation blocked %s: %s",
                    response.next_action.type,
                    ", ".join(validation.reasons),
                )
                self._consecutive_failures += 1
                s.consecutive_failures = self._consecutive_failures
                s.last_result = f"FAIL: blocked by coordinate validation ({', '.join(validation.reasons)})"
                if validation.retry_hint:
                    self._user_messages.append(f"HINT: {validation.retry_hint}")
                record = ActionRecord(
                    step=step,
                    timestamp=datetime.now().isoformat(),
                    action_type=response.next_action.type,
                    params=response.next_action.params.model_dump(exclude_none=True),
                    observation=response.observation,
                    reasoning=response.reasoning,
                    confidence=response.confidence,
                    result="blocked",
                    verification=", ".join(validation.reasons),
                    grounding={
                        "score": validation.score,
                        "reasons": validation.reasons,
                        "provider_assessments": validation.provider_assessments,
                    },
                )
                self._memory.record_action(record)
                self._grounding_telemetry.record_outcome(
                    task_id=s.task_id,
                    step=step,
                    action_type=response.next_action.type,
                    result="blocked",
                    success=False,
                    diff_ratio=None,
                    validation=validation,
                )
                self._save_checkpoint()
                return ExecutionResult(
                    success=False,
                    action_type=response.next_action.type,
                    error=f"coordinate validation blocked action: {', '.join(validation.reasons)}",
                )

            if response.next_action.type == "mouse_drag":
                logger.info(
                    "Coordinate validation approved %s from (%d, %d) to (%d, %d) score=%.2f",
                    response.next_action.type,
                    response.next_action.params.x1 if response.next_action.params.x1 is not None else -1,
                    response.next_action.params.y1 if response.next_action.params.y1 is not None else -1,
                    response.next_action.params.x2 if response.next_action.params.x2 is not None else -1,
                    response.next_action.params.y2 if response.next_action.params.y2 is not None else -1,
                    validation.score,
                )
            else:
                logger.info(
                    "Coordinate validation approved %s at (%d, %d) score=%.2f",
                    response.next_action.type,
                    response.next_action.params.x if response.next_action.params.x is not None else -1,
                    response.next_action.params.y if response.next_action.params.y is not None else -1,
                    validation.score,
                )

        s.next_action = (
            f"{response.next_action.type}({response.next_action.params.model_dump(exclude_none=True)})"
        )

        # Loop detection: check for repeated identical actions
        snapshot = StepSnapshot(
            action_type=response.next_action.type,
            params_hash=hash(str(response.next_action.params.model_dump(exclude_none=True))),
            observation_prefix=response.observation[:50],
        )
        if self._detect_loop(snapshot):
            logger.warning("Infinite loop detected! Injecting recovery context.")
            self._user_messages.append(
                "WARNING: You appear to be repeating the same action. "
                "This action has failed or had no effect multiple times. "
                "Try something completely different."
            )
        self._recent_actions.append(snapshot)

        # 4. ACT: Execute the action
        exec_result = self._executor.execute(response.next_action)
        s.actions_taken += 1

        result_text = exec_result.output[:100] if exec_result.output else exec_result.error[:100] if exec_result.error else ""
        s.last_result = f"{'OK' if exec_result.success else 'FAIL'}: {result_text}"
        logger.info("Result: %s", s.last_result)

        # Track consecutive failures
        if exec_result.success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        s.consecutive_failures = self._consecutive_failures

        # 5. VERIFY: Observe post-action changes and record the result
        diff_ratio: float | None = None
        if response.next_action.type not in ("wait", "task_complete", "shell_exec"):
            img = self._screen.capture_after_action(save=True)
            s.screenshot_path = f"agent-data/screenshots/screenshot_{self._screen.screenshot_count:05d}.png"

            # Screenshot diff: detect no-op actions
            if pre_action_img is not None and img is not None:
                diff_ratio = self._screen.diff_ratio(pre_action_img, img)
                if diff_ratio < self._settings.grounding.grounding_post_click_diff_threshold:
                    logger.warning(
                        "No-op detected: action %s produced <%.2f%% pixel change",
                        response.next_action.type, diff_ratio * 100,
                    )
                    self._user_messages.append(
                        f"HINT: Your last action ({response.next_action.type}) did not produce "
                        "any visible change on screen. The action may have missed its target "
                        "or had no effect. Consider a different approach."
                    )

        grounding_info = {}
        if validation is not None:
            grounding_info = {
                "score": validation.score,
                "reasons": validation.reasons,
                "pixel_point": list(validation.pixel_point) if validation.pixel_point is not None else None,
                "pixel_bbox": list(validation.pixel_bbox) if validation.pixel_bbox is not None else None,
                "pixel_point_end": list(validation.pixel_point_end) if validation.pixel_point_end is not None else None,
                "pixel_bbox_end": list(validation.pixel_bbox_end) if validation.pixel_bbox_end is not None else None,
                "provider_assessments": validation.provider_assessments,
            }
            if diff_ratio is not None:
                grounding_info["diff_ratio"] = diff_ratio

        record = ActionRecord(
            step=step,
            timestamp=datetime.now().isoformat(),
            action_type=response.next_action.type,
            params=response.next_action.params.model_dump(exclude_none=True),
            observation=response.observation,
            reasoning=response.reasoning,
            confidence=response.confidence,
            result="success" if exec_result.success else "failure",
            verification=exec_result.output or exec_result.error,
            grounding=grounding_info,
        )
        self._memory.record_action(record)
        if validation is not None:
            self._grounding_telemetry.record_outcome(
                task_id=s.task_id,
                step=step,
                action_type=response.next_action.type,
                result="success" if exec_result.success else "failure",
                success=exec_result.success,
                diff_ratio=diff_ratio,
                validation=validation,
            )

        self._maybe_queue_post_typing_verification_hint(
            task=task,
            action=response.next_action,
            exec_result=exec_result,
        )

        # Checkpoint: persist state so the task can be resumed
        self._save_checkpoint()

        return exec_result

    # ── Error recovery ───────────────────────────────────────

    def _detect_loop(self, current: StepSnapshot) -> bool:
        """Detect if the agent is stuck in a loop of identical actions."""
        if len(self._recent_actions) < _MAX_IDENTICAL_ACTIONS:
            return False

        recent = list(self._recent_actions)[-_MAX_IDENTICAL_ACTIONS:]
        return all(
            a.action_type == current.action_type and a.params_hash == current.params_hash
            for a in recent
        )

    # ── Helpers ───────────────────────────────────────────────

    def _infer_task_type(self, task: str) -> str | None:
        """Rough inference of task type for state collection."""
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["code", "코드", "코딩", "프로그래밍", "개발", "build", "test"]):
            return "coding"
        if any(kw in task_lower for kw in ["browse", "search", "브라우저", "검색", "웹"]):
            return "browsing"
        return None

    def _format_state(self, state) -> str:
        """Format SystemState into text for the prompt."""
        lines = [
            f"CWD: {state.cwd}",
            f"Foreground Window: {state.foreground_window}",
            f"Processes:\n{state.processes}",
        ]
        if state.git_status is not None:
            lines.append(f"Git Status:\n{state.git_status}")
        for key, val in state.extra.items():
            lines.append(f"{key}: {val}")
        return "\n\n".join(lines)

    def _format_capture_context(self, capture_context: CaptureContext) -> str:
        """Format screenshot metadata for the model prompt."""
        return (
            f"Screenshot size: {capture_context.capture_width}x{capture_context.capture_height}\n"
            f"Screen size: {capture_context.screen_width}x{capture_context.screen_height}\n"
            f"Screen origin: ({capture_context.screen_left}, {capture_context.screen_top})\n"
            f"Capture target: {capture_context.capture_target}\n"
            "Return mouse grounding as normalized coordinates in the 0.0-1.0 range relative "
            "to the screenshot."
        )

    def _maybe_queue_post_typing_verification_hint(
        self,
        *,
        task: str,
        action,
        exec_result: ExecutionResult,
    ) -> None:
        """Guide the model to verify visible text after simple typing tasks."""
        if not exec_result.success:
            return
        if action.type != "type_text":
            return

        task_lower = task.casefold()
        simple_markers = [
            "메모장", "notepad", "입력", "타이핑", "써봐", "write", "type", "텍스트",
        ]
        complex_markers = [
            "code", "코드", "vscode", "copilot", "build", "test", "git", "terminal",
            "powershell", "project", "프로젝트", "folder", "폴더", "directory", "디렉터리",
            "save", "저장", "browser", "브라우저", "chrome",
        ]
        if not any(marker in task_lower for marker in simple_markers):
            return
        if any(marker in task_lower for marker in complex_markers):
            return

        typed_text = (action.params.text or "").strip()
        if not typed_text:
            return

        self._user_messages.append(
            "HINT: Your last type_text action reported success. On the next step, first verify that the requested text is visibly present in the target app. If it is visible, return task_complete. If it is not visible, refocus the text area and try typing again."
        )

    def _maybe_refresh_grounding_providers(
        self,
        screenshot_bytes: bytes,
        capture_context: CaptureContext,
    ) -> None:
        """Optionally refresh external grounding provider files for the current screenshot."""
        self._maybe_refresh_gemini_grounding_provider(screenshot_bytes, capture_context)
        self._maybe_refresh_omniparser_provider(screenshot_bytes, capture_context)

    def _maybe_refresh_gemini_grounding_provider(
        self,
        screenshot_bytes: bytes,
        capture_context: CaptureContext,
    ) -> None:
        """Optionally refresh the Gemini OCR provider file for the current screenshot."""
        grounding = self._settings.grounding
        if not grounding.grounding_enable_gemini_ocr_provider:
            return
        if not grounding.grounding_auto_refresh_gemini_ocr:
            return
        if self._grounding_extractor is None:
            return

        try:
            self._agent_state.api_calls += 1
            screenshot_path = save_grounding_capture_image(
                self._settings.agent_data_dir,
                screenshot_bytes,
                timestamp=capture_context.timestamp,
            )
            payload = self._grounding_extractor.extract_provider_payload(
                image_bytes=screenshot_bytes,
                capture_context=capture_context,
                use_pro=self._settings.gemini.gemini_use_pro_for_vision,
                max_items=grounding.grounding_gemini_ocr_max_items,
                source_image_path=screenshot_path,
            )
            provider_path = write_grounding_provider_payload(
                self._settings.agent_data_dir / grounding.grounding_provider_inputs_subdir,
                payload,
                output_name=grounding.grounding_gemini_ocr_output_name,
            )
            logger.debug("Gemini OCR grounding provider refreshed: %s", provider_path)
        except Exception as exc:
            logger.warning("Gemini OCR grounding provider refresh failed: %s", exc)

    def _maybe_refresh_omniparser_provider(
        self,
        screenshot_bytes: bytes,
        capture_context: CaptureContext,
    ) -> None:
        """Optionally refresh the OmniParser provider file for the current screenshot."""
        grounding = self._settings.grounding
        if not grounding.grounding_enable_omniparser_runner:
            return
        if not grounding.grounding_auto_refresh_omniparser:
            return
        if self._omniparser_runner is None:
            return

        try:
            screenshot_path = save_grounding_capture_image(
                self._settings.agent_data_dir,
                screenshot_bytes,
                timestamp=capture_context.timestamp,
            )
            payload = self._omniparser_runner.extract_provider_payload(
                image_path=screenshot_path,
                capture_context=capture_context,
            )
            provider_path = write_grounding_provider_payload(
                self._settings.agent_data_dir / grounding.grounding_provider_inputs_subdir,
                payload,
                output_name=grounding.grounding_omniparser_output_name,
            )
            logger.debug("OmniParser grounding provider refreshed: %s", provider_path)
        except Exception as exc:
            logger.warning("OmniParser grounding provider refresh failed: %s", exc)

    def _save_checkpoint(self) -> None:
        """Persist the current runtime counters for resume support."""
        s = self._agent_state
        self._memory.save_checkpoint({
            "confidence": s.confidence,
            "consecutive_failures": self._consecutive_failures,
            "api_calls": s.api_calls,
            "actions_taken": s.actions_taken,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "total_tokens": s.total_tokens,
        })
