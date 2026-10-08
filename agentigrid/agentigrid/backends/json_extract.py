"""Robust JSON extraction from LLM response text."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger("agentigrid.backends.json_extract")

# Regex patterns
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _try_parse(text: str, label: str) -> Optional[dict]:
    """Attempt ``json.loads`` on *text*, returning the dict or None."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            logger.debug("JSON parsed successfully (%s)", label)
            return obj
        logger.debug("Parsed JSON is not a dict (%s): %s", label, type(obj).__name__)
    except (json.JSONDecodeError, ValueError):
        logger.debug("json.loads failed (%s)", label)
    return None


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]``."""
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def extract_json(text: str) -> tuple[Optional[dict], Optional[str]]:
    """Extract a JSON object from LLM response text.

    Tries multiple strategies in order:
    1. Direct ``json.loads`` on the full text.
    2. Strip markdown code fences and parse the inner content.
    3. Regex-extract ``{…}`` blocks and try each.
    4. Retry all strategies after removing trailing commas.

    Returns:
        Tuple of ``(parsed_dict, error_message)``. Exactly one is None.
    """
    if not text or not text.strip():
        return None, "Empty response text"

    stripped = text.strip()

    # --- Strategy 1: direct parse ---
    result = _try_parse(stripped, "direct")
    if result is not None:
        return result, None

    # --- Strategy 2: markdown fences ---
    for match in _FENCE_RE.finditer(stripped):
        inner = match.group(1).strip()
        result = _try_parse(inner, "fence")
        if result is not None:
            return result, None

    # --- Strategy 3: regex brace extraction ---
    for match in _BRACE_RE.finditer(stripped):
        result = _try_parse(match.group(0), "brace")
        if result is not None:
            return result, None

    # --- Retry all strategies with trailing-comma fix ---
    fixed = _fix_trailing_commas(stripped)
    if fixed != stripped:
        result = _try_parse(fixed, "direct+comma-fix")
        if result is not None:
            return result, None

        for match in _FENCE_RE.finditer(fixed):
            inner = _fix_trailing_commas(match.group(1).strip())
            result = _try_parse(inner, "fence+comma-fix")
            if result is not None:
                return result, None

        for match in _BRACE_RE.finditer(fixed):
            result = _try_parse(_fix_trailing_commas(match.group(0)), "brace+comma-fix")
            if result is not None:
                return result, None

    snippet = stripped[:120] + ("…" if len(stripped) > 120 else "")
    return None, f"No valid JSON object found in response: {snippet}"
