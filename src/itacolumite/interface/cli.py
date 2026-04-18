"""CLI interface – the user-facing command layer (native Windows)."""

from __future__ import annotations

import logging
import threading
import time

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
        lines.append(f"[bold]Confidence:[/bold] {state.confidence:.2f}")
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


@cli.command()
@click.argument("description")
@click.option("--no-live", is_flag=True, help="Disable Rich live display")
def task(description: str, no_live: bool) -> None:
    """Give the agent a task to perform.

    Example: itacolumite task "Create a Python Flask TODO API"
    """
    from itacolumite.core.agent import Agent

    console.print(BANNER)
    console.print(f"[bold]Task:[/bold] {description}\n")

    settings = get_settings()
    console.print(f"[dim]Control pipe: \\\\.\\pipe\\{settings.agent.control_pipe_name}[/dim]\n")

    agent = Agent()

    if no_live:
        # Simple mode: no live display, just logs
        try:
            agent.start()
            result = agent.run_task(description)
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
            r = agent.run_task(description)
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

    table.add_row("Shell", settings.agent.shell_executable)
    table.add_row("Max Steps", str(settings.agent.agent_max_steps))
    table.add_row("Control Pipe", f"\\\\.\\pipe\\{settings.agent.control_pipe_name}")
    table.add_row("DPI Awareness", settings.agent.dpi_awareness)
    table.add_row("Capture Target", settings.agent.capture_target)
    table.add_row("Gemini Model", settings.gemini.gemini_model)
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
