"""LLM instructions for router sessions (several tools, one goal)."""

from agentigrid.prompts.router.system_prompt import build_analysis_prompt, build_system_prompt

__all__ = ["build_analysis_prompt", "build_system_prompt"]
