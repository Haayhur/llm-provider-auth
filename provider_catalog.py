"""Shared provider metadata and model catalogs.

This module is the single source of truth for provider display metadata and
default model catalogs used by the orchestration API layer.
"""

from __future__ import annotations

from copy import deepcopy

# Shared provider display metadata for UI/status endpoints.
OAUTH_PROVIDER_METADATA = {
    "antigravity": {"name": "Antigravity (Google)", "type": "oauth"},
    "codex": {"name": "Codex (OpenAI)", "type": "oauth"},
    "copilot": {"name": "GitHub Copilot", "type": "oauth"},
}

# Detailed model catalogs for provider `/models` endpoints.
ANTIGRAVITY_ROUTE_MODELS = {
    "gemini": [
        {
            "id": "antigravity-gemini-3-flash",
            "name": "Gemini 3 Flash",
            "description": "Fast, efficient model with thinking",
        },
        {
            "id": "antigravity-gemini-3-pro-low",
            "name": "Gemini 3 Pro (Low)",
            "description": "Pro model with low thinking budget",
        },
        {
            "id": "antigravity-gemini-3-pro-high",
            "name": "Gemini 3 Pro (High)",
            "description": "Pro model with high thinking budget",
        },
    ],
    "claude": [
        {
            "id": "antigravity-claude-sonnet-4-6",
            "name": "Claude Sonnet 4.6",
            "description": "Claude Sonnet 4.6",
        },
        {
            "id": "antigravity-claude-opus-4-6-thinking-low",
            "name": "Claude Opus 4.6 (Thinking Low)",
            "description": "8K thinking budget",
        },
        {
            "id": "antigravity-claude-opus-4-6-thinking-max",
            "name": "Claude Opus 4.6 (Thinking Max)",
            "description": "32K thinking budget",
        },
    ],
}

CODEX_ROUTE_MODELS = {
    "gpt5": [
        {
            "id": "gpt-5.3-codex",
            "name": "GPT-5.3 Codex",
            "description": "Latest GPT-5 Codex model",
        },
        {
            "id": "gpt-5-codex",
            "name": "GPT-5 Codex",
            "description": "GPT-5 Codex family model",
        },
        {
            "id": "gpt-5.2-codex",
            "name": "GPT-5.2 Codex",
            "description": "GPT-5.2 optimized for code",
        },
        {
            "id": "gpt-5.1-codex",
            "name": "GPT-5.1 Codex",
            "description": "GPT-5.1 optimized for code",
        },
        {
            "id": "gpt-5.1-codex-max",
            "name": "GPT-5.1 Codex Max",
            "description": "GPT-5.1 Codex with max capabilities",
        },
        {
            "id": "gpt-5.1-codex-mini",
            "name": "GPT-5.1 Codex Mini",
            "description": "Lightweight GPT-5.1 Codex",
        },
        {
            "id": "codex-mini-latest",
            "name": "Codex Mini Latest",
            "description": "Alias for the latest Codex Mini model",
        },
        {
            "id": "gpt-5.2",
            "name": "GPT-5.2",
            "description": "GPT-5 base model retained by Codex OAuth flow",
        },
    ],
}

# Compact model catalogs for provider status views.
ANTIGRAVITY_PROFILE_MODELS = [
    {"id": "antigravity-gemini-3-flash", "name": "Gemini 3 Flash"},
    {"id": "antigravity-gemini-3-pro-low", "name": "Gemini 3 Pro (Low)"},
    {"id": "antigravity-gemini-3-pro-high", "name": "Gemini 3 Pro (High)"},
    {"id": "antigravity-claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
    {"id": "antigravity-claude-opus-4-6-thinking-max", "name": "Claude Opus 4.6 (Thinking)"},
]

CODEX_PROFILE_MODELS = [
    {"id": "gpt-5.3-codex", "name": "GPT-5.3 Codex"},
    {"id": "gpt-5-codex", "name": "GPT-5 Codex"},
    {"id": "gpt-5.2-codex", "name": "GPT-5.2 Codex"},
    {"id": "gpt-5.1-codex", "name": "GPT-5.1 Codex"},
    {"id": "gpt-5.1-codex-max", "name": "GPT-5.1 Codex Max"},
    {"id": "gpt-5.1-codex-mini", "name": "GPT-5.1 Codex Mini"},
    {"id": "codex-mini-latest", "name": "Codex Mini Latest"},
    {"id": "gpt-5.2", "name": "GPT-5.2"},
]


def get_provider_metadata() -> dict[str, dict[str, str]]:
    """Return provider display metadata (deep copy)."""
    return deepcopy(OAUTH_PROVIDER_METADATA)


def get_antigravity_route_models() -> dict[str, list[dict[str, str]]]:
    """Return Antigravity `/models` catalog (deep copy)."""
    return deepcopy(ANTIGRAVITY_ROUTE_MODELS)


def get_codex_route_models() -> dict[str, list[dict[str, str]]]:
    """Return Codex `/models` catalog (deep copy)."""
    return deepcopy(CODEX_ROUTE_MODELS)


def get_antigravity_profile_models() -> list[dict[str, str]]:
    """Return Antigravity compact profile model list (deep copy)."""
    return deepcopy(ANTIGRAVITY_PROFILE_MODELS)


def get_codex_profile_models() -> list[dict[str, str]]:
    """Return Codex compact profile model list (deep copy)."""
    return deepcopy(CODEX_PROFILE_MODELS)
