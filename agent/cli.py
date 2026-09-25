"""
agent/cli.py — Modern Typer & Rich CLI interface for the MCTS Agent.

Provides commands:
  - run: Execute closed-loop MCTS search with custom parameters.
  - demo: Run pre-configured task management demo.
  - interactive: Step-by-step human-in-the-loop search and execution.
  - visualize: Local agent web UI for prompts, live output, artifacts, and logs.

AI-Friendly & Headless Features:
  - --json: Clean machine-parseable JSON summary without ANSI codes on stdout.
  - --quiet / -q: Suppresses banners and progress output.
  - --no-color: Disables ANSI escape codes.
"""

from __future__ import annotations

import contextlib
from http.client import HTTPException
from http.cookies import SimpleCookie
import http.server
import ipaddress
import importlib.resources
import json
import os
import secrets
import sys
import threading
import webbrowser
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

# Ensure environment variables (.env) are loaded early for Typer option defaults
_repo_root = Path(__file__).resolve().parent.parent
_env_path = _repo_root / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path))
else:
    load_dotenv()

from agent.config import AgentConfig, load_config
from agent.mcts import (
    adapt_state,
    execute_single_action,
    review_action,
    run_closed_loop_agent,
    run_mcts,
)
from agent.node import Node
from agent.primitives import check_task_completion
from agent import updates

_DEMO_GOAL = (
    "Design a minimal REST API in Python (FastAPI) for a task management app: "
    "endpoints for creating, listing, updating, and deleting tasks, with in-memory storage."
)
_DEMO_INITIAL_STATE = (
    "We are starting from scratch. No code exists yet. "
    "The API must support: POST /tasks (create), GET /tasks (list all), "
    "PATCH /tasks/{id} (update), DELETE /tasks/{id} (delete). "
    "Storage is in-memory (a Python dict). No auth required."
)

app = typer.Typer(
    name="mcts-agent",
    help="Discriminative Monte Carlo Tree Search (MCTS) Agent powered by TypeSafe Jev System One Primitives and Gemini.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _setup_env(mock: bool) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
    else:
        load_dotenv()

    if not os.getenv("TYPESAFE_API_KEY") and os.getenv("API_KEY"):
        os.environ["TYPESAFE_API_KEY"] = os.environ["API_KEY"]

    if mock:
        os.environ["USE_MOCK_PRIMITIVES"] = "true"


def _create_consoles(no_color: bool) -> tuple[Console, Console]:
    if no_color:
        os.environ["NO_COLOR"] = "1"
    console = Console(no_color=no_color, highlight=not no_color)
    err_console = Console(stderr=True, no_color=no_color, highlight=not no_color)
    return console, err_console


@contextlib.contextmanager
def _redirect_stdout_for_mode(is_json: bool, is_quiet: bool):
    """
    Ensures stdout stays 100% clean JSON when --json is enabled, or quiet when --quiet is enabled.
    In JSON mode or quiet mode, intermediate stdout prints are redirected to devnull.
    """
    if is_json or is_quiet:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
    else:
        yield


def _render_banner(
    console: Console,
    goal: str,
    mock: bool,
    max_steps: int,
    iterations: int,
    workspace: str,
    proposal_model: Optional[str] = None,
) -> None:
    mode_text = (
        "[bold yellow]MOCK (No external APIs)[/bold yellow]"
        if mock
        else "[bold green]LIVE (Gemini & TypeSafe)[/bold green]"
    )
    content = (
        f"[bold cyan]Monte Carlo Tree Search Agent[/bold cyan] [dim]• Closed-Loop Reasoning[/dim]\n\n"
        f"[bold]Goal:[/bold] {goal}\n"
        f"[bold]Mode:[/bold] {mode_text}\n"
        f"[bold]Max Steps:[/bold] {max_steps}  |  [bold]Iterations/Step:[/bold] {iterations}\n"
        f"[bold]Workspace:[/bold] [dim]{workspace}[/dim]"
    )
    if proposal_model:
        content += f"\n[bold]Proposal Model:[/bold] [dim]{proposal_model}[/dim]"
    console.print(Panel(content, title="[bold cyan]MCTS Agent[/bold cyan]", border_style="cyan"))


def _render_summary(console: Console, summary: dict[str, Any]) -> None:
    is_completed = summary.get("completed", False)
    stop_reason = summary.get("stop_reason", "completed" if is_completed else "max_steps")
    status_text = {
        "completed": "[bold green]COMPLETED[/bold green]",
        "no_action_generated": "[bold yellow]NO ACTION GENERATED[/bold yellow]",
        "max_steps": "[bold yellow]MAX STEPS REACHED[/bold yellow]",
    }.get(stop_reason, "[bold yellow]STOPPED[/bold yellow]")

    console.print()
    console.print(
        Panel(
            f"[bold]Goal:[/bold] {summary.get('goal')}\n"
            f"[bold]Status:[/bold] {status_text}\n"
            f"[bold]Total Steps Run:[/bold] {summary.get('total_steps_run', 0)}\n"
            f"[bold]Final State:[/bold] {summary.get('final_state', '')[:140]}...",
            title="[bold cyan]Execution Summary[/bold cyan]",
            border_style="green" if is_completed else "yellow",
        )
    )

    steps = summary.get("steps", [])
    if steps:
        table = Table(title="Executed Plan Steps", border_style="dim", header_style="bold cyan")
        table.add_column("Step", justify="center", style="bold")
        table.add_column("Action", style="bright_white")
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("Visits", justify="right", style="magenta")
        table.add_column("Execution", justify="center")
        table.add_column("Step Log", style="dim")

        for s in steps:
            exec_info = s.get("execution", {})
            if not exec_info.get("success", False):
                exec_status = "[red]✗ Failed[/red]"
            elif not exec_info.get("verified", False):
                exec_status = "[yellow]? Unverified[/yellow]"
            else:
                exec_status = "[green]✓ Verified[/green]"
            log_name = os.path.basename(s.get("mcts_log", ""))
            table.add_row(
                str(s.get("step")),
                s.get("action", ""),
                f"{s.get('score', 0.0):.2f}/10",
                str(s.get("visits", 0)),
                exec_status,
                log_name,
            )
        console.print(table)

    summary_file = summary.get("summary_file")
    if summary_file:
        console.print(f"\n[dim]Complete Run Summary saved to:[/dim] [cyan]{summary_file}[/cyan]")
    console.print("[dim]Open the agent web UI with:[/dim] [bold cyan]mcts-agent visualize[/bold cyan]\n")


@app.callback()
def main_callback(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output clean machine-parseable JSON summary without ANSI escape codes.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Quiet mode: suppress informational messages and progress banners.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable ANSI color output.",
    ),
    no_update_check: bool = typer.Option(
        False,
        "--no-update-check",
        envvar="MCTS_AGENT_DISABLE_UPDATE_CHECK",
        help="Skip the once-daily PyPI update check.",
    ),
):
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output
    ctx.obj["quiet"] = quiet
    ctx.obj["no_color"] = no_color
    ctx.obj["no_update_check"] = no_update_check

    if no_update_check or ctx.invoked_subcommand in {None, "version", "update"}:
        return
    latest = updates.check_latest_quietly()
    if latest and updates.update_available(updates.installed_version(), latest):
        sys.stderr.write(
            f"mcts-agent {latest} available (installed {updates.installed_version()}). "
            "Run `mcts-agent update` to upgrade.\n"
        )


@app.command("version")
def show_version() -> None:
    """Show the installed version and the latest version on PyPI."""
    current = updates.installed_version()
    typer.echo(f"mcts-agent {current}")
    try:
        latest = updates.latest_version(force=True)
    except (OSError, HTTPException, updates.UpdateCheckUnavailable, ValueError, TypeError, KeyError) as exc:
        typer.echo(f"Latest: unavailable ({exc})")
        return
    typer.echo(f"Latest: {latest}")
    if updates.update_available(current, latest):
        typer.echo("Update available.\n\nRun:\n    mcts-agent update")
    elif updates.update_available(latest, current):
        typer.echo("Installed version is newer than the latest PyPI release.")
    else:
        typer.echo("Up to date.")


@app.command("update")
def update_package() -> None:
    """Explicitly upgrade mcts-agent from PyPI."""
    current = updates.installed_version()
    typer.echo(f"Current version: {current}")
    try:
        latest = updates.latest_version(force=True)
    except (OSError, HTTPException, updates.UpdateCheckUnavailable, ValueError, TypeError, KeyError) as exc:
        typer.echo(f"Unable to check PyPI: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Latest version:  {latest}")
    if not updates.update_available(current, latest):
        typer.echo("Already up to date.")
        return

    typer.echo(f"\nDownloading mcts-agent {latest}...")
    try:
        result = updates.perform_update(latest)
    except OSError as exc:
        typer.echo(f"Unable to run the package installer: {exc}", err=True)
        raise typer.Exit(code=1)
    if result.returncode != 0:
        typer.echo("Update failed. See installer output above.", err=True)
        raise typer.Exit(code=result.returncode)
    typer.echo("✓ Updated successfully")
    notes = updates.release_notes(latest)
    if notes:
        typer.echo("\nRelease notes:")
        typer.echo(notes)


@app.command()
def run(
    ctx: typer.Context,
    goal: Optional[str] = typer.Option(
        None, "--goal", "-g", help="Custom goal for MCTS agent to achieve."
    ),
    state: Optional[str] = typer.Option(
        None, "--state", "-s", help="Initial state / context description for search."
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Working directory for agent execution."
    ),
    planner: Optional[str] = typer.Option(
        None, "--planner", "-p", help="Planner harness (agy | pi | mock)."
    ),
    executor: Optional[str] = typer.Option(
        None, "--executor", "-e", help="Executor harness (agy | pi | mock)."
    ),
    proposal_model: str = typer.Option(
        os.getenv("AGY_PROPOSAL_MODEL", "gemini-3.8-flash-low"),
        "--proposal-model",
        envvar="AGY_PROPOSAL_MODEL",
        help="AI model for generating candidate action proposals (default: gemini-3.8-flash-low).",
    ),
    max_steps: Optional[int] = typer.Option(
        int(os.getenv("MCTS_MAX_STEPS")) if os.getenv("MCTS_MAX_STEPS") else None,
        "--max-steps",
        help="Max steps in execute-review-adapt loop (default: dynamic mode via JEV/Noul assessment).",
    ),
    iterations: int = typer.Option(
        int(os.getenv("MCTS_ITERATIONS", "10")),
        "--iterations",
        "-n",
        help="MCTS iterations per step (default: 10).",
    ),
    sim_depth: Optional[int] = typer.Option(
        None,
        "--sim-depth",
        help="Fixed simulation lookahead depth (default: dynamic prime selection).",
    ),
    actions: Optional[int] = typer.Option(
        None,
        "--actions",
        "-a",
        help="Fixed actions per node (default: dynamic prime selection).",
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Run in mock mode without calling external APIs."
    ),
    no_early_stop: bool = typer.Option(
        False,
        "--no-early-stop",
        help="Disable Noul-based completion early stopping.",
    ),
    no_execute: bool = typer.Option(
        False,
        "--no-execute",
        help="Skip executing actions in the workspace.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output clean machine-parseable JSON summary without ANSI escape codes.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Quiet mode: suppress informational messages and progress banners.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable ANSI color output.",
    ),
):
    """Run closed-loop MCTS agent to plan and execute steps towards a goal."""
    is_json = json_output or ctx.obj.get("json", False)
    is_quiet = quiet or ctx.obj.get("quiet", False)
    is_no_color = no_color or ctx.obj.get("no_color", False)
    console, err_console = _create_consoles(is_no_color)

    _setup_env(mock=mock)
    if proposal_model:
        os.environ["AGY_PROPOSAL_MODEL"] = proposal_model
    mock_mode = os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes")

    if not mock_mode and not os.getenv("TYPESAFE_API_KEY"):
        if not is_json and not is_quiet:
            err_console.print(
                "[yellow]WARNING: TYPESAFE_API_KEY not set. Pass --mock to skip live API calls.[/yellow]"
            )

    active_goal = goal or _DEMO_GOAL
    active_state = state or (
        _DEMO_INITIAL_STATE if not goal else f"Starting state for goal: {active_goal}"
    )
    early_stop_noul = not no_early_stop
    execute_plan = not no_execute
    try:
        current_cwd = os.getcwd()
    except Exception:
        current_cwd = "."
    resolved_workspace = workspace or current_cwd

    # ``demo`` calls this command function directly. Typer's option defaults are
    # OptionInfo objects in that path, rather than the CLI's resolved ``None``.
    if isinstance(planner, typer.models.OptionInfo):
        planner = None
    if isinstance(executor, typer.models.OptionInfo):
        executor = None

    config_overrides: dict[str, Any] = {}
    if planner:
        config_overrides["planner_provider"] = planner
    if executor:
        config_overrides["executor_provider"] = executor

    cfg = load_config(cli_overrides=config_overrides)

    if not is_json and not is_quiet:
        _render_banner(
            console=console,
            goal=active_goal,
            mock=mock_mode,
            max_steps=max_steps,
            iterations=iterations,
            workspace=resolved_workspace,
            proposal_model=cfg.planner_model,
        )

    try:
        with _redirect_stdout_for_mode(is_json=is_json, is_quiet=is_quiet):
            summary = run_closed_loop_agent(
                goal=active_goal,
                initial_state=active_state,
                max_steps=max_steps,
                iterations_per_step=iterations,
                actions_per_node=actions,
                simulation_depth=sim_depth,
                early_stop_noul=early_stop_noul,
                execute=execute_plan,
                workspace_dir=resolved_workspace,
                config=cfg,
            )
    except Exception as exc:
        if is_json:
            err_dict = {
                "error": str(exc),
                "status": "failed",
                "goal": active_goal,
            }
            sys.stdout.write(json.dumps(err_dict, indent=2) + "\n")
            sys.stdout.flush()
            raise typer.Exit(code=1)
        else:
            err_console.print(f"[bold red]Execution error:[/bold red] {exc}")
            raise typer.Exit(code=1)

    summary_file = Path(__file__).resolve().parent.parent / "logs" / f"run_{summary.get('run_id')}_summary.json"
    if summary_file.exists():
        summary["summary_file"] = str(summary_file.resolve())

    if is_json:
        sys.stdout.write(json.dumps(summary, indent=2, default=str) + "\n")
        sys.stdout.flush()
    elif is_quiet:
        status_label = {
            "completed": "COMPLETED",
            "no_action_generated": "NO_ACTION_GENERATED",
            "max_steps": "MAX_STEPS_REACHED",
        }.get(summary.get("stop_reason", "completed" if summary.get("completed") else "max_steps"), "STOPPED")
        console.print(f"Goal: {active_goal}")
        console.print(f"Status: {status_label} | Steps: {summary.get('total_steps_run', 0)}")
        if "summary_file" in summary:
            console.print(f"Summary: {summary['summary_file']}")
    else:
        _render_summary(console=console, summary=summary)


@app.command()
def demo(
    ctx: typer.Context,
    iterations: int = typer.Option(
        int(os.getenv("MCTS_ITERATIONS", "10")),
        "--iterations",
        "-n",
        help="MCTS iterations per step (default: 10).",
    ),
    max_steps: int = typer.Option(
        int(os.getenv("MCTS_MAX_STEPS", "5")),
        "--max-steps",
        help="Max steps in the execute-review-adapt loop (default: 5).",
    ),
    actions: Optional[int] = typer.Option(
        None,
        "--actions",
        "-a",
        help="Fixed actions per node (default: dynamic prime selection).",
    ),
    sim_depth: Optional[int] = typer.Option(
        None,
        "--sim-depth",
        help="Fixed simulation lookahead depth (default: dynamic prime selection).",
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Working directory for agent execution."
    ),
    proposal_model: str = typer.Option(
        os.getenv("AGY_PROPOSAL_MODEL", "gemini-3.8-flash-low"),
        "--proposal-model",
        envvar="AGY_PROPOSAL_MODEL",
        help="AI model for generating candidate action proposals (default: gemini-3.8-flash-low).",
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Run in mock mode without calling external APIs."
    ),
    no_early_stop: bool = typer.Option(
        False,
        "--no-early-stop",
        help="Disable Noul-based completion early stopping.",
    ),
    no_execute: bool = typer.Option(
        False,
        "--no-execute",
        help="Skip executing actions in the workspace.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output clean machine-parseable JSON summary without ANSI escape codes.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Quiet mode: suppress informational messages and progress banners.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable ANSI color output.",
    ),
):
    """Run the pre-configured FastAPI Task Management demo."""
    run(
        ctx=ctx,
        goal=_DEMO_GOAL,
        state=_DEMO_INITIAL_STATE,
        workspace=workspace,
        max_steps=max_steps,
        iterations=iterations,
        sim_depth=sim_depth,
        actions=actions,
        mock=mock,
        no_early_stop=no_early_stop,
        no_execute=no_execute,
        proposal_model=proposal_model,
        json_output=json_output,
        quiet=quiet,
        no_color=no_color,
    )


@app.command()
def interactive(
    ctx: typer.Context,
    iterations: int = typer.Option(
        int(os.getenv("MCTS_ITERATIONS", "10")),
        "--iterations",
        "-n",
        help="MCTS iterations per step (default: 10).",
    ),
    max_steps: int = typer.Option(
        int(os.getenv("MCTS_MAX_STEPS", "5")),
        "--max-steps",
        help="Max steps in the interactive loop (default: 5).",
    ),
    actions: Optional[int] = typer.Option(
        None,
        "--actions",
        "-a",
        help="Fixed actions per node (default: dynamic prime selection).",
    ),
    sim_depth: Optional[int] = typer.Option(
        None,
        "--sim-depth",
        help="Fixed simulation lookahead depth (default: dynamic prime selection).",
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Run in mock mode without calling external APIs."
    ),
    no_early_stop: bool = typer.Option(
        False,
        "--no-early-stop",
        help="Disable Noul-based completion early stopping.",
    ),
    no_execute: bool = typer.Option(
        False,
        "--no-execute",
        help="Skip executing actions in the workspace.",
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Working directory for agent execution."
    ),
    proposal_model: str = typer.Option(
        os.getenv("AGY_PROPOSAL_MODEL", "gemini-3.8-flash-low"),
        "--proposal-model",
        envvar="AGY_PROPOSAL_MODEL",
        help="AI model for generating candidate action proposals (default: gemini-3.8-flash-low).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output clean machine-parseable JSON summary without ANSI escape codes.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Quiet mode: suppress informational messages and progress banners.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable ANSI color output.",
    ),
):
    """Interactive step-by-step MCTS agent with human-in-the-loop control."""
    is_json = json_output or (ctx.obj and ctx.obj.get("json", False))
    is_quiet = quiet or (ctx.obj and ctx.obj.get("quiet", False))
    if is_json:
        raise typer.BadParameter(
            "Interactive mode requires a terminal and cannot be run with --json.",
            param_hint="--json",
        )
    if is_quiet:
        raise typer.BadParameter(
            "Interactive mode requires a terminal and cannot be run with --quiet.",
            param_hint="--quiet",
        )

    is_no_color = no_color or (ctx.obj and ctx.obj.get("no_color", False))
    console, _ = _create_consoles(is_no_color)
    _setup_env(mock=mock)
    if proposal_model:
        os.environ["AGY_PROPOSAL_MODEL"] = proposal_model
    mock_mode = os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes")

    if not mock_mode and not os.getenv("TYPESAFE_API_KEY"):
        console.print(
            "[yellow]WARNING: TYPESAFE_API_KEY not set. Pass --mock to skip live calls.[/yellow]"
        )

    console.print(
        Panel(
            "[bold cyan]Interactive Closed-Loop MCTS Agent[/bold cyan]\n"
            "[dim]Human-in-the-loop: inspect search recommendations, choose to execute or override.[/dim]",
            title="[bold green]Interactive Mode[/bold green]",
            border_style="cyan",
        )
    )

    goal = typer.prompt("Enter your goal").strip()
    if not goal or goal.lower() == "quit":
        console.print("[yellow]Exiting interactive session.[/yellow]")
        return

    initial_state = typer.prompt(
        "Enter initial context/state (press Enter for default)",
        default="",
        show_default=False,
    ).strip()
    if not initial_state:
        initial_state = f"Starting state for goal: {goal}"

    current_state = initial_state
    try:
        current_cwd = os.getcwd()
    except Exception:
        current_cwd = "."
    target_workspace = workspace or current_cwd
    early_stop = not no_early_stop
    execute_plan = not no_execute

    for step in range(1, max_steps + 1):
        console.print()
        console.rule(f"[bold cyan]Turn {step} / {max_steps}[/bold cyan]")
        console.print(f"[dim]Current State:[/dim] {current_state[:120]}...")

        root = Node(state=current_state)
        best, log_path = run_mcts(
            root,
            goal=goal,
            iterations=iterations,
            actions_per_node=actions,
            simulation_depth=sim_depth,
            early_stop_noul=early_stop,
            step=step,
        )

        chosen_action = best.action_taken
        if not chosen_action:
            console.print("[yellow]No action proposed by MCTS. Ending session.[/yellow]")
            break

        console.print(
            Panel(
                f"[bold green]Recommended Action:[/bold green] {chosen_action}\n"
                f"[bold]Average Score:[/bold] {best.average_value:.2f}/10  |  [bold]Visits:[/bold] {best.visits}\n"
                f"[dim]Step Log:[/dim] [cyan]{os.path.abspath(log_path)}[/cyan]",
                title="[bold]MCTS Decision[/bold]",
                border_style="green",
            )
        )

        user_choice = typer.prompt(
            "[E]xecute action, [S]kip execution, [N]ew state manually, or [Q]uit?",
            default="E",
        ).strip().lower()

        if user_choice in ("q", "quit"):
            console.print("[yellow]Interactive session ended by user.[/yellow]")
            break

        if user_choice == "n":
            custom_state = typer.prompt("Enter new state description").strip()
            if custom_state:
                current_state = custom_state
            continue

        if user_choice == "s" or not execute_plan:
            console.print("[dim]Skipping execution of this action.[/dim]")
            exec_res = {
                "success": True,
                "skipped": True,
                "stdout": "Execution skipped by user.",
                "stderr": "",
                "returncode": 0,
            }
        else:
            console.print("[dim]Executing action in workspace...[/dim]")
            exec_res = execute_single_action(
                chosen_action, goal, current_state, workspace_dir=target_workspace
            )
            if exec_res.get("success"):
                console.print("[green]Execution succeeded.[/green]")
            else:
                console.print(
                    f"[red]Execution reported failure (exit code {exec_res.get('returncode')})[/red]"
                )

        observation = review_action(chosen_action, exec_res, workspace_dir=target_workspace)
        current_state = adapt_state(
            current_state, step=step, action=chosen_action, observation=observation
        )

        if early_stop:
            is_done, conf = check_task_completion(goal, current_state)
            if is_done:
                console.print(
                    Panel(
                        f"[bold green]✓ Goal verified COMPLETED by Noul (confidence={conf:.3f})![/bold green]",
                        border_style="green",
                    )
                )
                break


@app.command()
def visualize(
    ctx: typer.Context,
    port: int = typer.Option(
        8000, "--port", "-p", help="Port for the local agent web UI."
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host interface to bind server to."
    ),
    browser: bool = typer.Option(
        True,
        "--browser/--no-browser",
        help="Open the web app in browser automatically.",
    ),
    directory: Optional[Path] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Workspace directory served by the app, including logs and image artifacts (default: current directory).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output server details in JSON format.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Quiet mode: suppress server logs.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable ANSI color output.",
    ),
):
    """Start the local MCTS Agent Web UI for prompts, live output, artifacts, and logs."""
    is_json = json_output or ctx.obj.get("json", False)
    is_quiet = quiet or ctx.obj.get("quiet", False)
    is_no_color = no_color or ctx.obj.get("no_color", False)
    console, _ = _create_consoles(is_no_color)

    visualizer_path: Optional[Path] = None
    try:
        resource = importlib.resources.files("agent") / "visualizer.html"
        candidate = Path(str(resource))
        if candidate.is_file():
            visualizer_path = candidate
    except Exception:
        visualizer_path = None

    if directory:
        serve_dir = directory.resolve()
    else:
        serve_dir = Path.cwd().resolve()

    if visualizer_path is None or not visualizer_path.exists():
        if is_json:
            sys.stdout.write(
                json.dumps({"error": f"visualizer.html not found in {serve_dir}"}) + "\n"
            )
            raise typer.Exit(code=1)
        console.print(
            f"[yellow]Warning: visualizer.html not found in {serve_dir}. Serving directory anyway.[/yellow]"
        )

    url = f"http://{host}:{port}/"

    class LiveRun:
        """One local run plus its SSE-readable event history."""
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []
            self.condition = threading.Condition()
            self.running = False

        def publish(self, message: dict[str, Any]) -> None:
            with self.condition:
                self.messages.append(message)
                self.condition.notify_all()

        def begin(self, metadata: dict[str, Any]) -> bool:
            with self.condition:
                if self.running:
                    return False
                self.messages = [{"kind": "meta", "meta": metadata}]
                self.running = True
                self.condition.notify_all()
                return True

        def finish(self, status: str, **details: Any) -> None:
            with self.condition:
                self.running = False
                self.messages.append({"kind": "status", "status": status, **details})
                self.condition.notify_all()

    live_run = LiveRun()
    session_token = secrets.token_urlsafe(32)

    class VisualizerHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def do_GET(self):
            requested_path = self.path.partition("?")[0]
            if requested_path == "/api/events":
                if not self._has_session_token():
                    self.send_error(403)
                    return
                self._stream_events()
                return
            if requested_path == "/api/logs":
                if not self._has_session_token():
                    self.send_error(403)
                    return
                self._json_response(200, {"logs": self._list_logs()})
                return
            if requested_path in ("/", ""):
                self.path = "/visualizer.html"
                requested_path = "/visualizer.html"
            # Always serve the resolved visualizer asset. A project-root
            # visualizer.html may be a stale copy; letting SimpleHTTPRequestHandler
            # serve it here silently bypasses the current packaged UI.
            if requested_path == "/visualizer.html" and visualizer_path and visualizer_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                content = visualizer_path.read_bytes()
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header(
                    "Set-Cookie",
                    f"MCTS-Session={session_token}; Path=/; HttpOnly; SameSite=Strict",
                )
                self.end_headers()
                self.wfile.write(content)
                return
            return super().do_GET()

        def do_POST(self):
            if self.path.partition("?")[0] != "/api/run":
                self.send_error(404)
                return
            if not self._has_session_token():
                self.send_error(403)
                return
            origin = self.headers.get("Origin", "")
            host_header = self.headers.get("Host", "")
            try:
                parsed_origin = urlsplit(origin)
                parsed_host = urlsplit(f"http://{host_header}")
                origin_name = (parsed_origin.hostname or "").lower()
                parsed_host_name = (parsed_host.hostname or "").lower()
                is_localhost = origin_name == parsed_host_name == "localhost"
                if not is_localhost:
                    origin_host = ipaddress.ip_address(origin_name)
                origin_valid = (
                    parsed_origin.scheme == "http"
                    and (parsed_origin.port or 80) == port
                    and parsed_origin.netloc == parsed_host.netloc
                    and (is_localhost or parsed_host_name == str(origin_host))
                )
            except ValueError:
                origin_valid = False
            if not origin_valid:
                self.send_error(403, "Request origin does not match this UI")
                return
            if self.headers.get_content_type() != "application/json":
                self.send_error(415, "Content-Type must be application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                goal = str(payload.get("goal", "")).strip()
                if not goal:
                    raise ValueError("A goal is required.")
                context = str(payload.get("context", "")).strip() or "Work on the requested goal in the selected workspace."
                steps = max(1, min(int(payload.get("steps", 1)), 10))
                width = max(1, min(int(payload.get("width", 3)), 5))
                depth = max(1, min(int(payload.get("depth", 2)), 4))
                mock = bool(payload.get("mock", False))
                execute = bool(payload.get("execute", False))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json_response(400, {"error": f"Invalid run request: {exc}"})
                return
            metadata = {"goal": goal, "initial_state": context, "iterations": width ** depth, "live": True}
            if not live_run.begin(metadata):
                self._json_response(409, {"error": "A run is already in progress."})
                return

            def run_live_agent() -> None:
                prior_mock = os.environ.get("USE_MOCK_PRIMITIVES")
                output = None

                class RunOutput:
                    """Forward complete console lines to the live output panel."""
                    def __init__(self) -> None:
                        self.pending = ""

                    def write(self, chunk: str) -> int:
                        self.pending += chunk
                        while "\n" in self.pending:
                            line, self.pending = self.pending.split("\n", 1)
                            live_run.publish({"kind": "text", "text": line + "\n"})
                        return len(chunk)

                    def flush(self) -> None:
                        if self.pending:
                            live_run.publish({"kind": "text", "text": self.pending})
                            self.pending = ""

                try:
                    cfg = load_config()
                    if mock:
                        cfg = replace(cfg, planner_provider="mock", executor_provider="mock")
                        os.environ["USE_MOCK_PRIMITIVES"] = "true"
                    output = RunOutput()
                    with contextlib.redirect_stdout(output):
                        summary = run_closed_loop_agent(
                            goal=goal, initial_state=context, max_steps=steps,
                            actions_per_node=width, expansion_depth=depth, execute=execute,
                            workspace_dir=str(serve_dir), log_dir=str(serve_dir / "logs"), config=cfg,
                            event_sink=lambda event: live_run.publish({"kind": "event", "event": event}),
                        )
                    output.flush()
                    image_files = sorted({
                        path
                        for step in summary.get("steps", [])
                        for path in step.get("execution", {}).get("changed_files", [])
                        if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
                    })
                    live_run.publish({"kind": "artifact", "images": image_files})
                    live_run.finish("complete", summary=summary)
                except Exception as exc:
                    if output is not None:
                        output.flush()
                    live_run.finish("error", error=str(exc))
                finally:
                    if mock:
                        if prior_mock is None:
                            os.environ.pop("USE_MOCK_PRIMITIVES", None)
                        else:
                            os.environ["USE_MOCK_PRIMITIVES"] = prior_mock

            threading.Thread(target=run_live_agent, daemon=True, name="mcts-live-run").start()
            self._json_response(202, {"status": "started"})

        def _has_session_token(self) -> bool:
            cookies = SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
            except Exception:
                return False
            supplied = cookies.get("MCTS-Session")
            return supplied is not None and secrets.compare_digest(supplied.value, session_token)

        def _list_logs(self) -> list[dict[str, Any]]:
            logs_dir = Path(serve_dir) / "logs"
            if not logs_dir.is_dir():
                return []
            entries = []
            for path in sorted(logs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    entries.append({
                        "name": path.name,
                        "path": f"logs/{path.name}",
                        "goal": payload.get("meta", {}).get("goal") or payload.get("goal", ""),
                        "updated": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes"),
                    })
                except (OSError, json.JSONDecodeError):
                    continue
            return entries

        def _json_response(self, code: int, body: dict[str, Any]) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def _stream_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            cursor = 0
            try:
                while True:
                    with live_run.condition:
                        if cursor >= len(live_run.messages):
                            live_run.condition.wait(timeout=1)
                        pending = live_run.messages[cursor:]
                        cursor = len(live_run.messages)
                    for message in pending:
                        self.wfile.write(b"data: " + json.dumps(message).encode("utf-8") + b"\n\n")
                    if pending:
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def log_message(self, format, *args):
            if is_quiet or is_json:
                return
            sys.stderr.write(f"[HTTP] {format % args}\n")

    try:
        httpd = http.server.ThreadingHTTPServer((host, port), VisualizerHandler)
    except OSError as exc:
        if is_json:
            sys.stdout.write(
                json.dumps({"error": f"Failed to bind port {port}: {exc}"}) + "\n"
            )
            raise typer.Exit(code=1)
        console.print(f"[bold red]Failed to start server on {host}:{port}:[/bold red] {exc}")
        console.print(f"[dim]Try specifying a different port with: --port {port + 1}[/dim]")
        raise typer.Exit(code=1)

    if is_json:
        sys.stdout.write(
            json.dumps(
                {
                    "status": "running",
                    "url": url,
                    "host": host,
                    "port": port,
                    "directory": str(serve_dir),
                },
                indent=2,
            )
            + "\n"
        )
        sys.stdout.flush()
    elif not is_quiet:
        console.print(
            Panel(
                f"[bold green]MCTS Agent Web UI[/bold green]\n\n"
                f"[bold]URL:[/bold] [link={url}]{url}[/link]\n"
                f"[bold]Serving Directory:[/bold] [dim]{serve_dir}[/dim]\n"
                f"[bold]Logs Directory:[/bold] [dim]{serve_dir / 'logs'}[/dim]\n\n"
                f"[dim]Prompt a run, follow its output, preview image artifacts, and browse saved logs.[/dim]\n"
                f"[bold yellow]Press Ctrl+C to stop the server.[/bold yellow]",
                title="[bold cyan]Agent Web UI[/bold cyan]",
                border_style="cyan",
            )
        )

    if browser and not is_json:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        with httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        if not is_json and not is_quiet:
            console.print("\n[yellow]Agent web UI server stopped.[/yellow]")


from agent.research.cli import register_research_command

register_research_command(app)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
