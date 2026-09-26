"""CLI registration kept separate from the legacy execution and web UI commands."""

import contextlib
import json
import sys
from pathlib import Path
from typing import Optional

import typer

from agent import updates


def register_research_command(app: typer.Typer) -> None:
    @app.command("research")
    def research(
        ctx: typer.Context,
        query: Optional[str] = typer.Option(None, "--query", "-q"),
        workspace: Optional[Path] = typer.Option(None, "--workspace", "-w"),
        resume: Optional[Path] = typer.Option(None, help="Run directory or checkpoint.json to resume"),
        brief: Optional[Path] = typer.Option(None, help="Optional reviewed research brief JSON"),
        max_steps: int = typer.Option(5, min=1, help="Additional investigations for this invocation"),
        width: int = typer.Option(2, min=1, max=8, help="New-run search width"),
        depth: int = typer.Option(2, min=1, max=5, help="New-run search depth"),
        max_nodes: int = typer.Option(64, min=1, max=10000, help="New-run tree size limit"),
        planner: Optional[str] = typer.Option(None),
        executor: Optional[str] = typer.Option(None),
        system_one: Optional[str] = typer.Option(None, "--system-one"),
        mock: bool = typer.Option(False, help="Offline orchestration demo; never validates a discovery"),
        json_output: bool = typer.Option(False, "--json"),
    ):
        """Investigate an open question with an evidence notebook and resumable steps."""
        from agent.config import load_config
        from .models import ResearchBrief, ResearchSettings
        from .runner import run_research

        is_json = json_output or bool((ctx.obj or {}).get("json"))
        updates.show_update_notice(
            disabled=bool((ctx.obj or {}).get("no_update_check")),
            quiet=bool((ctx.obj or {}).get("quiet")),
        )
        try:
            if resume and (planner or executor or system_one):
                raise ValueError("Resume preserves the saved provider configuration")
            # All provider chatter goes to stderr; stdout stays machine-readable.
            with contextlib.redirect_stdout(sys.stderr):
                cfg = None if resume else load_config(cli_overrides={
                    k: v for k, v in {"planner_provider": planner, "executor_provider": executor,
                                     "system_one_provider": system_one}.items() if v
                })
                research_brief = ResearchBrief.model_validate_json(brief.read_text(encoding="utf-8")) if brief else None
                limits = None if resume else ResearchSettings(width=width, depth=depth, max_nodes=max_nodes)
                state, run_dir = run_research(
                    query, workspace=str(workspace) if workspace else None,
                    resume=str(resume) if resume else None, config=cfg, settings=limits,
                    max_steps=max_steps, brief=research_brief, mock=mock,
                )
            result = {"status": state.status, "run_id": state.run_id, "query": state.query,
                      "steps": len(state.steps), "mock": state.mock,
                      "checkpoint": str(run_dir / "checkpoint.json"),
                      "report": str(run_dir / "report.md"), "error": state.error}
        except (OSError, ValueError) as exc:
            result = {"status": "failed", "error": str(exc)}
        if is_json:
            typer.echo(json.dumps(result, indent=2))
        else:
            typer.echo(f"Research status: {result['status']}")
            if result.get("report"):
                typer.echo(f"Report: {result['report']}\nCheckpoint: {result['checkpoint']}")
            if result.get("error"):
                typer.echo(result["error"], err=True)
        if result["status"] in ("failed", "blocked"):
            raise typer.Exit(code=1)
