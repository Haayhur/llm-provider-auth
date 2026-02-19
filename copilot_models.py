"""
Helper utilities for listing Copilot models via the Copilot CLI SDK.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any

try:
    from .copilot_auth import CopilotAuth
except Exception:  # pragma: no cover
    from copilot_auth import CopilotAuth  # type: ignore


DEFAULT_COPILOT_MODELS = [
    {"id": "claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "description": "Claude Sonnet 4.5 via Copilot"},
    {"id": "claude-haiku-4.5", "name": "Claude Haiku 4.5", "description": "Claude Haiku 4.5 via Copilot"},
    {"id": "claude-opus-4.5", "name": "Claude Opus 4.5", "description": "Claude Opus 4.5 via Copilot"},
    {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "description": "Claude Sonnet 4 via Copilot"},
    {"id": "gpt-5.2-codex", "name": "GPT-5.2-Codex", "description": "GPT-5.2-Codex via Copilot"},
    {"id": "gpt-5.1-codex-max", "name": "GPT-5.1-Codex-Max", "description": "GPT-5.1-Codex-Max via Copilot"},
    {"id": "gpt-5.1-codex", "name": "GPT-5.1-Codex", "description": "GPT-5.1-Codex via Copilot"},
    {"id": "gpt-5.2", "name": "GPT-5.2", "description": "GPT-5.2 via Copilot"},
    {"id": "gpt-5.1", "name": "GPT-5.1", "description": "GPT-5.1 via Copilot"},
    {"id": "gpt-5", "name": "GPT-5", "description": "GPT-5 via Copilot"},
    {"id": "gpt-5.1-codex-mini", "name": "GPT-5.1-Codex-Mini", "description": "GPT-5.1-Codex-Mini via Copilot"},
    {"id": "gpt-5-mini", "name": "GPT-5 mini", "description": "GPT-5 mini via Copilot"},
    {"id": "gpt-4.1", "name": "GPT-4.1", "description": "GPT-4.1 via Copilot"},
    {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro (Preview)", "description": "Gemini 3 Pro (Preview) via Copilot"},
]

_MODEL_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]+")
_MODEL_STOPWORDS = {"model", "models", "available", "current", "default"}


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    if value is None:
        return ""
    return str(value)


def _looks_like_model_id(token: str) -> bool:
    if not token or len(token) < 2:
        return False
    if token.lower() in _MODEL_STOPWORDS:
        return False
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]+$", token):
        return False
    if not any(ch.isalpha() for ch in token):
        return False
    if not (any(ch.isdigit() for ch in token) or "-" in token or "_" in token or "." in token):
        return False
    return True


def parse_copilot_models(text: str) -> list[str]:
    if not text:
        return []

    candidates: list[str] = []

    # Prefer explicit model IDs shown in backticks.
    candidates.extend(re.findall(r"`([^`]+)`", text))

    # Then parse bullet-list style output.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            candidates.append(line[2:].strip())
        elif line.startswith("* "):
            candidates.append(line[2:].strip())

    # Fallback: free-form token scan.
    candidates.extend(_MODEL_TOKEN_RE.findall(text))

    results: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        normalized = str(token or "").strip().strip("`'\"()[]{}.,;:")
        if not normalized:
            continue
        # Bullet lines can include trailing descriptions.
        if " " in normalized:
            normalized = normalized.split()[0]
        if not normalized:
            continue
        if not _looks_like_model_id(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
    return results


def model_id_to_name(model_id: str) -> str:
    value = model_id.replace("_", "-").strip()
    if not value:
        return model_id
    parts = [p for p in value.split("-") if p]
    if not parts:
        return model_id

    head = parts[0]
    if head.lower() == "gpt":
        head = "GPT"
    else:
        head = head.capitalize()

    tail_parts: list[str] = []
    for part in parts[1:]:
        if part.replace(".", "").isdigit():
            tail_parts.append(part)
        else:
            tail_parts.append(part.capitalize())

    if head == "GPT" and tail_parts:
        first_tail = tail_parts[0]
        if first_tail.replace(".", "").isdigit():
            rest = " ".join(tail_parts[1:])
            return f"GPT-{first_tail}" + (f" {rest}" if rest else "")

    return f"{head} {' '.join(tail_parts)}".strip()


def _resolve_cli_options(
    *,
    cli_path: str | None = None,
    cli_url: str | None = None,
    use_stdio: bool | None = None,
    log_level: str | None = None,
) -> tuple[str | None, str | None, bool | None, str]:
    resolved_cli_url = cli_url or os.environ.get("COPILOT_CLI_URL")
    resolved_cli_path = _resolve_cli_path(cli_path or os.environ.get("COPILOT_CLI_PATH"))

    resolved_use_stdio = use_stdio
    if resolved_use_stdio is None:
        env_value = (os.environ.get("COPILOT_USE_STDIO") or "").strip().lower()
        if env_value:
            resolved_use_stdio = env_value in {"1", "true", "yes", "on"}

    resolved_log_level = log_level or os.environ.get("COPILOT_LOG_LEVEL") or "info"
    return resolved_cli_url, resolved_cli_path, resolved_use_stdio, resolved_log_level


def _resolve_cli_path(configured_path: str | None) -> str | None:
    raw = (configured_path or "").strip()
    if raw:
        expanded = os.path.expanduser(raw)
        if any(sep in expanded for sep in ("/", "\\")):
            return expanded
        located = shutil.which(expanded)
        return located or expanded
    # Unset: try PATH command first, then allow SDK bundled binary fallback.
    return shutil.which("copilot")


def _cli_exists(cli_path: str | None) -> bool:
    if not cli_path:
        return False
    return os.path.exists(os.path.expanduser(cli_path))


def _load_copilot_client():
    try:
        from copilot import CopilotClient  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "github-copilot-sdk is required for Copilot model listing. "
            "Install it with `pip install github-copilot-sdk`."
        ) from exc
    return CopilotClient


async def fetch_copilot_models(
    auth: CopilotAuth | None,
    *,
    cli_path: str | None = None,
    cli_url: str | None = None,
    use_stdio: bool | None = None,
    log_level: str | None = None,
    timeout: float = 30.0,
    require_auth: bool = True,
) -> list[dict[str, str]]:
    if require_auth:
        if not auth or not auth.access_token:
            return []
        if auth.is_expired():
            return []
    elif auth and auth.is_expired():
        return []

    resolved_cli_url, resolved_cli_path, resolved_use_stdio, resolved_log_level = _resolve_cli_options(
        cli_path=cli_path,
        cli_url=cli_url,
        use_stdio=use_stdio,
        log_level=log_level,
    )

    if not resolved_cli_url and resolved_cli_path and not _cli_exists(resolved_cli_path):
        raise FileNotFoundError(f"Copilot CLI not found: {resolved_cli_path}")

    env = os.environ.copy()
    if auth and auth.access_token:
        env["GH_TOKEN"] = auth.access_token
        env["GITHUB_TOKEN"] = auth.access_token
    if auth and auth.enterprise_url:
        env["COPILOT_DOMAIN"] = auth.enterprise_url

    options: dict[str, Any] = {"log_level": resolved_log_level, "env": env}
    if resolved_cli_url:
        options["cli_url"] = resolved_cli_url
    else:
        if resolved_cli_path:
            options["cli_path"] = resolved_cli_path
        if resolved_use_stdio is not None:
            options["use_stdio"] = resolved_use_stdio

    CopilotClient = _load_copilot_client()
    client = CopilotClient(options)
    session = None
    try:
        await client.start()
        session = await client.create_session()
        event = await session.send_and_wait({"prompt": "/model"}, timeout=timeout)
        content = ""
        if event and hasattr(event.data, "content"):
            content = _coerce_text(event.data.content)
        model_ids = parse_copilot_models(content)
        if not model_ids:
            return DEFAULT_COPILOT_MODELS
        results: list[dict[str, str]] = []
        for model_id in model_ids:
            name = model_id_to_name(model_id)
            results.append(
                {
                    "id": model_id,
                    "name": name or model_id,
                    "description": f"{name or model_id} via Copilot",
                }
            )
        return results
    finally:
        if session:
            try:
                await session.destroy()
            except Exception:
                pass
        try:
            await client.stop()
        except Exception:
            pass
