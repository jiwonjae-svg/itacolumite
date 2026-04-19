"""CLI interface – the user-facing command layer (native Windows)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from itacolumite.ai.gemini_client import GeminiClient, is_placeholder_api_key
from itacolumite.config.settings import get_settings
from itacolumite.interface.logger import setup_logging

console = Console()
logger = logging.getLogger(__name__)

BANNER = r"""
[bold cyan]
 ╦╔╦╗╔═╗╔═╗╔═╗╦  ╦ ╦╔╦╗╦╔╦╗╔═╗
 ║ ║ ╠═╣║  ║ ║║  ║ ║║║║║ ║ ║╣
 ╩ ╩ ╩ ╩╚═╝╚═╝╩═╝╚═╝╩ ╩╩ ╩ ╚═╝
[/bold cyan]
[dim]Autonomous computer-use agent on native Windows[/dim]
"""


def _build_agent_panel(state) -> Panel:
    """Build a Rich Panel showing real-time agent state."""
    elapsed = time.time() - state.start_time if state.start_time else 0
    minutes, seconds = divmod(int(elapsed), 60)

    status_str = "[bold green]Running[/bold green]"
    if state.paused:
        status_str = "[bold yellow]Paused[/bold yellow]"
    elif not state.running:
        status_str = "[dim]Stopped[/dim]"

    lines = [
        f"[bold]Task:[/bold] {state.task}",
        f"[bold]Step:[/bold] {state.step} / {state.max_steps}  |  "
        f"Status: {status_str}  |  "
        f"Elapsed: {minutes}m {seconds:02d}s",
        f"[bold]API calls:[/bold] {state.api_calls}  |  "
        f"[bold]Actions:[/bold] {state.actions_taken}  |  "
        f"[bold]Failures:[/bold] {state.consecutive_failures}",
        f"[bold]Tokens:[/bold] prompt={state.prompt_tokens:,}  "
        f"completion={state.completion_tokens:,}  "
        f"total={state.total_tokens:,}",
        "",
    ]

    if state.observation:
        obs = state.observation[:200] + "..." if len(state.observation) > 200 else state.observation
        lines.append(f"[cyan bold]Observation:[/cyan bold]\n  {obs}")
        lines.append("")

    if state.reasoning:
        rea = state.reasoning[:200] + "..." if len(state.reasoning) > 200 else state.reasoning
        lines.append(f"[yellow bold]Reasoning:[/yellow bold]\n  {rea}")
        lines.append("")

    if state.plan:
        lines.append("[magenta bold]Plan:[/magenta bold]")
        for i, step_text in enumerate(state.plan[:5], 1):
            lines.append(f"  {i}. {step_text}")
        lines.append("")

    if state.next_action:
        lines.append(f"[green bold]Next Action:[/green bold] {state.next_action}")
        conf_color = "red" if state.confidence < 0.3 else "yellow" if state.confidence < 0.6 else "green"
        lines.append(f"[bold]Confidence:[/bold] [{conf_color}]{state.confidence:.2f}[/{conf_color}]")
        if state.current_model:
            is_pro = "pro" in state.current_model.lower()
            model_tag = "[red bold]Pro ✦[/red bold]" if is_pro else "[dim]Flash[/dim]"
            lines.append(f"[bold]Model:[/bold] {model_tag} [dim]{state.current_model}[/dim]")
        lines.append("")

    if state.last_result:
        lines.append(f"[dim]Last Result: {state.last_result}[/dim]")

    if state.screenshot_path:
        lines.append(f"[dim]Screenshot: {state.screenshot_path}[/dim]")

    content = "\n".join(lines)
    return Panel(
        content,
        title="[bold]Itacolumite Agent[/bold]",
        subtitle=f"[dim]Control: \\\\.\\pipe\\{get_settings().agent.control_pipe_name}[/dim]",
        border_style="cyan",
        padding=(1, 2),
    )


@click.group(invoke_without_command=True)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """Itacolumite – AI agent that controls a native Windows desktop."""
    setup_logging("DEBUG" if debug else get_settings().agent.agent_log_level)

    if ctx.invoked_subcommand is None:
        console.print(BANNER)
        console.print("Run [bold]itacolumite --help[/bold] for available commands.\n")


def _ensure_gemini_ready(api_key: str | None = None) -> None:
    """Validate Gemini configuration before starting a task."""
    configured_key = api_key if api_key is not None else get_settings().gemini.gemini_api_key

    if is_placeholder_api_key(configured_key):
        raise click.ClickException(
            "GEMINI_API_KEY is not configured. Edit .env and set a valid API key before running a task."
        )

    try:
        GeminiClient().validate_api_access()
    except Exception as exc:
        raise click.ClickException(f"Gemini API validation failed: {exc}") from exc


@cli.command()
@click.argument("description", required=False, default="")
@click.option("--no-live", is_flag=True, help="Disable Rich live display")
@click.option("--resume", "resume_id", default=None, help="Resume a previously interrupted task by its task-ID (or 'latest')")
def task(description: str, no_live: bool, resume_id: str | None) -> None:
    """Give the agent a task to perform.

    Example: itacolumite task "Create a Python Flask TODO API"
    Resume:  itacolumite task --resume latest
    """
    from itacolumite.core.memory import Memory

    # Resolve resume
    actual_resume_id: str | None = None
    if resume_id:
        if resume_id == "latest":
            actual_resume_id = Memory.latest_checkpoint_id()
            if actual_resume_id is None:
                raise click.ClickException("No checkpoint found to resume.")
        else:
            actual_resume_id = resume_id
        console.print(BANNER)
        console.print(f"[bold]Resuming task:[/bold] {actual_resume_id}\n")
    else:
        if not description:
            raise click.ClickException("Task description is required (or use --resume).")
        console.print(BANNER)
        console.print(f"[bold]Task:[/bold] {description}\n")

    settings = get_settings()
    console.print(f"[dim]Control pipe: \\\\.\\pipe\\{settings.agent.control_pipe_name}[/dim]\n")

    _ensure_gemini_ready(settings.gemini.gemini_api_key)

    from itacolumite.core.agent import Agent

    agent = Agent()

    if no_live:
        # Simple mode: no live display, just logs
        try:
            agent.start()
            result = agent.run_task(description, resume_task_id=actual_resume_id)
            console.print(Panel.fit(
                f"[bold]Result:[/bold] {result}",
                title="Task Complete",
                border_style="green",
            ))
        except KeyboardInterrupt:
            console.print("\n[yellow]Task interrupted by user.[/yellow]")
        finally:
            agent.stop()
        return

    # Live display mode: run agent in a thread, update display in main thread
    result_holder: list[str] = []
    error_holder: list[Exception] = []

    def _run_agent() -> None:
        try:
            agent.start()
            r = agent.run_task(description, resume_task_id=actual_resume_id)
            result_holder.append(r)
        except KeyboardInterrupt:
            result_holder.append("interrupted_by_user")
        except Exception as e:
            error_holder.append(e)
            result_holder.append(f"error: {e}")
        finally:
            agent.stop()

    agent_thread = threading.Thread(target=_run_agent, daemon=True)

    try:
        with Live(
            _build_agent_panel(agent.state),
            console=console,
            refresh_per_second=2,
            transient=False,
        ) as live:
            agent_thread.start()

            while agent_thread.is_alive():
                live.update(_build_agent_panel(agent.state))
                time.sleep(0.5)

            # Final update
            live.update(_build_agent_panel(agent.state))

    except KeyboardInterrupt:
        agent.stop()
        agent_thread.join(timeout=5)
        console.print("\n[yellow]Task interrupted by user.[/yellow]")
        return

    agent_thread.join(timeout=5)

    if result_holder:
        result = result_holder[0]
        color = "green" if "error" not in result else "red"
        console.print(Panel.fit(
            f"[bold]Result:[/bold] {result}",
            title="Task Complete",
            border_style=color,
        ))

    if error_holder:
        console.print(f"[red]Error: {error_holder[0]}[/red]")


@cli.command()
def status() -> None:
    """Show agent configuration."""
    settings = get_settings()

    table = Table(title="Agent Settings")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    auto = settings.gemini.gemini_auto_upgrade
    threshold = settings.gemini.gemini_auto_upgrade_threshold

    table.add_row("Shell", settings.agent.shell_executable)
    table.add_row("Max Steps", str(settings.agent.agent_max_steps))
    table.add_row("Control Pipe", f"\\\\.\\pipe\\{settings.agent.control_pipe_name}")
    table.add_row("DPI Awareness", settings.agent.dpi_awareness)
    table.add_row("Capture Target", settings.agent.capture_target)
    table.add_row("Gemini Fast Model", settings.gemini.gemini_model_fast)
    table.add_row("Gemini Pro Model", settings.gemini.gemini_model_pro)
    table.add_row(
        "Auto Upgrade",
        f"[green]enabled[/green] (conf<{threshold} or failures≥2)" if auto else "[dim]disabled[/dim]",
    )
    table.add_row("Screenshot Delay", f"{settings.agent.screenshot_delay_ms}ms")
    table.add_row("Action Delay", f"{settings.agent.action_delay_ms}ms")

    console.print(table)


@cli.command()
def config() -> None:
    """Show current .env configuration."""
    status.invoke(click.Context(status))


@cli.command()
def screenshot() -> None:
    """Capture the current screen and save to screenshot.png."""
    from itacolumite.perception.screen import ScreenCapture

    screen = ScreenCapture()
    img = screen.capture()
    img.save("screenshot.png")
    console.print(f"[green]✓ Screenshot saved to screenshot.png ({img.size[0]}x{img.size[1]})[/green]")


@cli.group()
def grounding() -> None:
    """Grounding utilities and telemetry reports."""
    pass


@grounding.command(name="extract-text")
@click.option("--output-name", default=None, help="Provider JSON file name under agent-data/grounding/providers")
@click.option("--pro/--fast", "use_pro", default=True, help="Use the Pro or Fast Gemini vision model")
def grounding_extract_text(output_name: str | None, use_pro: bool) -> None:
    """Capture the current screen and generate a grounding provider JSON file."""
    settings = get_settings()
    _ensure_gemini_ready(settings.gemini.gemini_api_key)
    provider_path, item_count, screenshot_path = _capture_grounding_text_provider(output_name, use_pro=use_pro)
    console.print("[green]✓ Grounding provider saved[/green]")
    click.echo(f"Provider path: {provider_path}")
    click.echo(f"Screenshot path: {screenshot_path}")
    click.echo(f"Anchors: {item_count}")


@grounding.command(name="run-omniparser")
@click.option("--output-name", default=None, help="Provider JSON file name under agent-data/grounding/providers")
def grounding_run_omniparser(output_name: str | None) -> None:
    """Capture the current screen and generate an OmniParser provider JSON file."""
    provider_path, item_count, screenshot_path = _capture_omniparser_provider(output_name)
    console.print("[green]✓ OmniParser provider saved[/green]")
    click.echo(f"Provider path: {provider_path}")
    click.echo(f"Screenshot path: {screenshot_path}")
    click.echo(f"Anchors: {item_count}")


@grounding.command(name="report")
@click.option(
    "--events-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional path to a validation_events.jsonl file",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional HTML report output path",
)
def grounding_report(events_path: Path | None, output_path: Path | None) -> None:
    """Summarize grounding validation telemetry and write an HTML report."""
    report_path, summary = _build_grounding_report(events_path, output_path)
    _print_grounding_summary(summary)
    console.print("[green]✓ Grounding report saved[/green]")
    click.echo(f"Report path: {report_path}")


@cli.group()
def control() -> None:
    """Send control commands to a running agent (via Named Pipe)."""
    pass


@control.command()
def pause() -> None:
    """Pause the running agent."""
    _send_control("pause")


@control.command()
def resume() -> None:
    """Resume the paused agent."""
    _send_control("resume")


@control.command(name="stop")
def control_stop() -> None:
    """Stop the running agent."""
    _send_control("stop")


@control.command()
@click.argument("message")
def send(message: str) -> None:
    """Send a text instruction to the running agent."""
    _send_control(f"send:{message}")


def _send_control(command: str) -> None:
    """Named Pipe로 제어 명령 전송."""
    import win32file

    pipe_name = f"\\\\.\\pipe\\{get_settings().agent.control_pipe_name}"
    try:
        handle = win32file.CreateFile(
            pipe_name,
            win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None,
        )
        win32file.WriteFile(handle, command.encode("utf-8"))
        win32file.CloseHandle(handle)
        console.print(f"[green]✓ Sent: {command}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to connect to pipe: {e}[/red]")
        console.print("[dim]Is the agent running?[/dim]")


def _capture_grounding_text_provider(
    output_name: str | None,
    *,
    use_pro: bool,
) -> tuple[Path, int, Path]:
    """Capture the current screen and write a provider file for grounding support."""
    from itacolumite.ai.gemini_client import GeminiClient
    from itacolumite.core.grounding_capture import (
        GeminiGroundingExtractor,
        save_grounding_capture_image,
        write_grounding_provider_payload,
    )
    from itacolumite.perception.screen import ScreenCapture

    settings = get_settings()
    screen = ScreenCapture()
    screenshot_bytes, capture_context = screen.capture_bytes_with_context()
    screenshot_path = save_grounding_capture_image(
        settings.agent_data_dir,
        screenshot_bytes,
        timestamp=capture_context.timestamp,
    )
    extractor = GeminiGroundingExtractor(GeminiClient())
    payload = extractor.extract_provider_payload(
        image_bytes=screenshot_bytes,
        capture_context=capture_context,
        use_pro=use_pro,
        max_items=settings.grounding.grounding_gemini_ocr_max_items,
        source_image_path=screenshot_path,
    )
    provider_path = write_grounding_provider_payload(
        settings.agent_data_dir / settings.grounding.grounding_provider_inputs_subdir,
        payload,
        output_name=output_name or settings.grounding.grounding_gemini_ocr_output_name,
    )
    return provider_path, len(payload.get("items") or []), screenshot_path


def _capture_omniparser_provider(output_name: str | None) -> tuple[Path, int, Path]:
    """Capture the current screen and write an OmniParser provider file."""
    from itacolumite.core.grounding_capture import (
        save_grounding_capture_image,
        write_grounding_provider_payload,
    )
    from itacolumite.core.omniparser_runner import OmniParserRunner
    from itacolumite.perception.screen import ScreenCapture

    settings = get_settings()
    runner = OmniParserRunner.from_settings(settings)
    if not runner.is_configured:
        raise click.ClickException("GROUNDING_OMNIPARSER_COMMAND is not configured")

    screen = ScreenCapture()
    screenshot_bytes, capture_context = screen.capture_bytes_with_context()
    screenshot_path = save_grounding_capture_image(
        settings.agent_data_dir,
        screenshot_bytes,
        timestamp=capture_context.timestamp,
    )
    payload = runner.extract_provider_payload(
        image_path=screenshot_path,
        capture_context=capture_context,
    )
    provider_path = write_grounding_provider_payload(
        settings.agent_data_dir / settings.grounding.grounding_provider_inputs_subdir,
        payload,
        output_name=output_name or settings.grounding.grounding_omniparser_output_name,
    )
    return provider_path, len(payload.get("items") or []), screenshot_path


def _build_grounding_report(
    events_path: Path | None,
    output_path: Path | None,
) -> tuple[Path, object]:
    """Load telemetry, render an HTML report, and return the output path and summary."""
    from itacolumite.core.grounding_report import (
        load_grounding_events,
        render_grounding_report_html,
        summarize_grounding_events,
        write_grounding_report,
    )

    settings = get_settings()
    resolved_events = events_path or (settings.agent_data_dir / "grounding" / "validation_events.jsonl")
    if not resolved_events.exists():
        raise click.ClickException(f"Grounding events file not found: {resolved_events}")

    summary = summarize_grounding_events(
        load_grounding_events(resolved_events),
        events_path=resolved_events,
    )
    resolved_output = output_path or (
        settings.agent_data_dir / settings.grounding.grounding_reports_subdir / "grounding_report.html"
    )
    write_grounding_report(resolved_output, render_grounding_report_html(summary))
    return resolved_output, summary


def _print_grounding_summary(summary: object) -> None:
    """Render a concise console summary for grounding telemetry."""
    metrics = Table(title="Grounding Telemetry")
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Value", style="white")
    metrics.add_row("Events", str(summary.total_events))
    metrics.add_row(
        "Validations",
        f"{summary.total_validations} total / {summary.approved_validations} approved / {summary.blocked_validations} blocked",
    )
    metrics.add_row("Approval rate", f"{summary.approval_rate:.1%}")
    metrics.add_row(
        "Outcomes",
        f"{summary.total_outcomes} total / {summary.successful_outcomes} success",
    )
    metrics.add_row(
        "Outcome success rate",
        f"{summary.success_rate:.1%}" if summary.success_rate is not None else "n/a",
    )
    metrics.add_row(
        "Average score",
        f"{summary.average_score:.3f}" if summary.average_score is not None else "n/a",
    )
    metrics.add_row(
        "Average diff ratio",
        f"{summary.average_diff_ratio:.4f}" if summary.average_diff_ratio is not None else "n/a",
    )
    console.print(metrics)

    if summary.reason_counts:
        reasons = Table(title="Top Grounding Reasons")
        reasons.add_column("Reason", style="magenta")
        reasons.add_column("Count", justify="right")
        for label, count in summary.reason_counts:
            reasons.add_row(label, str(count))
        console.print(reasons)
