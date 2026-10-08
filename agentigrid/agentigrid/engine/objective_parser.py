"""LLM-based objective extraction from natural language goals and steering directives.

The LLM parses user intent into structured objective definitions that get
registered in the ObjectiveRegistry.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("agentigrid.engine.objective_parser")


def build_objective_extraction_prompt(
    text: str,
    available_metrics: list[str],
    context: str = "initial_goal",
) -> tuple[str, str]:
    """Build prompt pair for LLM-based objective extraction.

    Args:
        text: The natural language text to extract objectives from
            (initial goal or steering directive).
        available_metrics: List of metric names with known extractors.
        context: "initial_goal" or "steering_directive".

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = (
        "You are analyzing a power grid optimization request to identify "
        "what objectives should be tracked. Extract structured objectives "
        "from the text. Respond ONLY with a JSON object, no other text."
    )

    user = (
        f"Text to analyze ({context}):\n"
        f'"{text}"\n\n'
        f"Available metric names (use these exact names when possible):\n"
        f"{json.dumps(available_metrics, indent=2)}\n\n"
        f"Extract objectives as a JSON object:\n"
        f'{{\n'
        f'  "objectives": [\n'
        f'    {{\n'
        f'      "name": "<metric name from the list above, or a custom descriptive name>",\n'
        f'      "direction": "<minimize | maximize | constraint>",\n'
        f'      "threshold": <number or null, for constraint-type objectives>,\n'
        f'      "priority": "<primary | secondary | watch>"\n'
        f'    }}\n'
        f'  ]\n'
        f'}}\n\n'
        f"Rules:\n"
        f"- Use metric names from the available list when they match the intent.\n"
        f"- Every goal has at least one primary objective.\n"
        f"- Constraints mentioned (like 'keep voltages above 0.95') become constraint-type objectives.\n"
        f"- If the text mentions monitoring something without optimizing it, use priority='watch'.\n"
        f"- For simple single-objective goals (e.g. 'minimize cost'), return just one objective.\n"
        f"- Only return objectives that are clearly stated or strongly implied.\n"
    )

    return system, user


def parse_objective_extraction(
    response_text: str,
) -> list[dict[str, Any]] | None:
    """Parse the LLM's objective extraction response.

    Args:
        response_text: Raw LLM response text.

    Returns:
        List of objective dicts, or None if parsing fails.
    """
    # Try ```json ... ``` fenced block first
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not match:
        # Try bare JSON with "objectives" key
        match = re.search(r'\{\s*"objectives"\s*:\s*\[.*?\]\s*\}', response_text, re.DOTALL)

    if not match:
        # Last resort: try the entire response as JSON
        try:
            data = json.loads(response_text.strip())
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not find objectives JSON in response")
            return None
    else:
        json_str = match.group(1) if match.lastindex else match.group(0)
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse objectives JSON: %s", exc)
            return None

    objectives = data.get("objectives", [])
    if not isinstance(objectives, list):
        logger.warning("'objectives' is not a list")
        return None

    # Validate each objective
    valid = []
    for obj in objectives:
        name = obj.get("name")
        direction = obj.get("direction", "minimize")
        if not name:
            continue
        if direction not in ("minimize", "maximize", "constraint"):
            direction = "minimize"
        priority = obj.get("priority", "primary")
        if priority not in ("primary", "secondary", "watch"):
            priority = "primary"
        valid.append({
            "name": name,
            "direction": direction,
            "threshold": obj.get("threshold"),
            "priority": priority,
        })

    return valid if valid else None
