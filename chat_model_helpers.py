"""Shared helpers for Antigravity chat model configuration."""

from __future__ import annotations

from typing import Any, Literal

import httpx

try:
    from .auth import AntigravityAuth
    from . import constants
except Exception:  # pragma: no cover
    from auth import AntigravityAuth  # type: ignore
    import constants  # type: ignore


def _retry_after_seconds(response: httpx.Response, *, default: float) -> float:
    """Best-effort parsing of server-provided backoff hints."""
    ra = (response.headers.get("retry-after") or "").strip()
    if ra:
        try:
            # Most commonly: integer seconds.
            seconds = float(ra)
            if seconds >= 0:
                return seconds
        except Exception:
            pass
    return default


def resolve_model_name(model: str) -> str:
    """Resolve user-facing model names to Antigravity API model IDs."""
    lower = model.strip()
    # Handle antigravity- prefix
    if lower.startswith(constants.ANTIGRAVITY_MODEL_PREFIX):
        mapped = constants.MODEL_MAPPINGS.get(lower)
        if mapped:
            return mapped
        base = lower[len(constants.ANTIGRAVITY_MODEL_PREFIX):]
        return constants.MODEL_MAPPINGS.get(base, base)
    return constants.MODEL_MAPPINGS.get(lower, lower)


def _env_project_override() -> str | None:
    """Return a non-default project override from env, if any."""
    env_project = constants.ANTIGRAVITY_PROJECT_ID
    if not env_project:
        return None
    if env_project == constants.ANTIGRAVITY_DEFAULT_PROJECT_ID:
        return None
    return env_project


def _resolve_auth_project_id(auth: AntigravityAuth | None) -> str | None:
    if not auth:
        return None
    return auth.project_id or auth.managed_project_id


def is_claude_model(model: str) -> bool:
    return "claude" in model.lower()


def is_thinking_model(model: str) -> bool:
    lower = model.lower()
    return "thinking" in lower or "gemini-3" in lower


def get_thinking_config(model: str) -> dict[str, Any] | None:
    """Get thinking configuration for a model."""
    lower = model.lower()

    if "claude" in lower and "thinking" in lower:
        if "low" in lower:
            budget = constants.THINKING_BUDGETS["low"]
        elif "high" in lower or "max" in lower:
            budget = constants.THINKING_BUDGETS["high"]
        else:
            budget = constants.THINKING_BUDGETS["medium"]

        return {
            "thinking_budget": budget,
            "include_thoughts": True,
        }

    if "gemini-3" in lower:
        if "flash" in lower:
            level = "minimal"
        elif "high" in lower:
            level = "high"
        else:
            level = "low"
        return {
            "includeThoughts": True,
            "thinkingLevel": level,
        }

    return None


def get_header_style(model: str) -> Literal["antigravity", "gemini-cli"]:
    """Determine which header style to use based on model."""
    lower = model.lower()

    # Antigravity quota models
    if lower.startswith("antigravity-") or "claude" in lower:
        return "antigravity"

    # Gemini CLI quota models
    if "preview" in lower or lower.startswith("gemini-2.") or lower.startswith("gemini-3-"):
        return "gemini-cli"

    # Default to antigravity for gemini-3
    if "gemini-3" in lower:
        return "antigravity"

    return "gemini-cli"
