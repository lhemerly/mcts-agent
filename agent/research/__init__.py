"""Modular open-ended investigation with evidence-aware validation."""

from .models import ResearchBrief, ResearchSettings, ResearchState
from .runner import run_research

__all__ = ["ResearchBrief", "ResearchSettings", "ResearchState", "run_research"]
