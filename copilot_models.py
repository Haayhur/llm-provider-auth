"""
Helper utilities for listing Copilot models via the Copilot CLI SDK.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any
from urllib.parse import urlparse

import httpx

import httpx

try:
    from .copilot_auth import CopilotAuth
except Exception:  # pragma: no cover
    from copilot_auth import CopilotAuth  # type: ignore


DEFAULT_COPILOT_MODELS = [
    {"id": "claude-haiku-4.5", "name": "Claude Haiku 4.5", "description": "Claude Haiku 4.5 via Copilot"},
    {"id": "claude-opus-41", "name": "Claude Opus 4.1", "description": "Claude Opus 4.1 via Copilot"},
    {"id": "claude-opus-4.5", "name": "Claude Opus 4.5", "description": "Claude Opus 4.5 via Copilot"},
    {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "description": "Claude Sonnet 4 via Copilot"},
    {"id": "claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "description": "Claude Sonnet 4.5 via Copilot"},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "description": "Gemini 2.5 Pro via Copilot"},
    {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash (Preview)", "description": "Gemini 3 Flash (Preview) via Copilot"},
    {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro (Preview)", "description": "Gemini 3 Pro (Preview) via Copilot"},
    {"id": "gpt-4.1", "name": "GPT-4.1", "description": "GPT-4.1 via Copilot"},
    {"id": "gpt-4o", "name": "GPT-4o", "description": "GPT-4o via Copilot"},
    {"id": "gpt-5", "name": "GPT-5", "description": "GPT-5 via Copilot"},
    {"id": "gpt-5-mini", "name": "GPT-5 mini", "description": "GPT-5 mini via Copilot"},
    {"id": "gpt-5.1", "name": "GPT-5.1", "description": "GPT-5.1 via Copilot"},
    {"id": "gpt-5.1-codex", "name": "GPT-5.1-Codex", "description": "GPT-5.1-Codex via Copilot"},
    {"id": "gpt-5.1-codex-max", "name": "GPT-5.1-Codex-Max", "description": "GPT-5.1-Codex-Max via Copilot"},
    {"id": "gpt-5.1-codex-mini", "name": "GPT-5.1-Codex-Mini", "description": "GPT-5.1-Codex-Mini via Copilot"},
    {"id": "gpt-5.2", "name": "GPT-5.2", "description": "GPT-5.2 via Copilot"},
    {"id": "gpt-5.2-codex", "name": "GPT-5.2-Codex", "description": "GPT-5.2-Codex via Copilot"},
    {"id": "grok-code-fast-1", "name": "Grok Code Fast 1", "description": "Grok Code Fast 1 via Copilot"},
]


def _normalize_domain(value: str | None) -> str:
    if not value:
        return ""
    cleaned = str(value).strip()
    if cleaned.startswith("http://"):
        cleaned = cleaned[len("http://") :]
    if cleaned.startswith("https://"):
        cleaned = cleaned[len("https://") :]
    cleaned = cleaned.split("/", 1)[0]
    return cleaned.rstrip("/")


def _copilot_api_base(enterprise_url: str | None) -> str:
    domain = _normalize_domain(enterprise_url)
    if domain and domain != "github.com":
        return f"https://copilot-api.{domain}"
    return "https://api.githubcopilot.com"


def _remote_model_name(model_id: str, payload: dict[str, Any]) -> str:
    name = payload.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return model_id_to_name(model_id)


def _remote_model_description(model_id: str, payload: dict[str, Any]) -> str:
    family = payload.get("capabilities", {}).get("family") if isinstance(payload.get("capabilities"), dict) else None
    if isinstance(family, str) and family.strip():
        return f"{_remote_model_name(model_id, payload)} via Copilot ({family.strip()})"
    return f"{_remote_model_name(model_id, payload)} via Copilot"


async def fetch_copilot_api_models(
    auth: CopilotAuth | None,
    *,
    timeout: float = 5.0,
) -> list[dict[str, str]]:
    if not auth or not auth.access_token:
        return []
    if auth.is_expired():
        return []

    base_url = _copilot_api_base(auth.enterprise_url)
    headers = {
        "Authorization": f"Bearer {auth.access_token}",
        "Accept": "application/json",
        "User-Agent": "co-scientist/copilot",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(f"{base_url}/models", headers=headers)
    if not response.is_success:
        raise ValueError(f"Failed to fetch Copilot models: {response.status_code}")
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    models: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("model_picker_enabled") is False:
            continue
        policy = item.get("policy")
        if isinstance(policy, dict) and policy.get("state") == "disabled":
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(
            {
                "id": model_id,
                "name": _remote_model_name(model_id, item),
                "description": _remote_model_description(model_id, item),
            }
        )
    return models

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


def _copilot_api_base(enterprise_url: str | None = None) -> str:
    raw = (enterprise_url or "").strip()
    if not raw:
        return "https://api.githubcopilot.com"

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    host = host.strip().rstrip("/")
    if not host or host == "github.com":
        return "https://api.githubcopilot.com"
    return f"https://copilot-api.{host}"


def _copilot_model_enabled(item: dict[str, Any]) -> bool:
    if item.get("model_picker_enabled") is False:
        return False
    policy = item.get("policy")
    if isinstance(policy, dict) and str(policy.get("state") or "").lower() == "disabled":
        return False
    return True


def _format_copilot_api_model(item: dict[str, Any]) -> dict[str, str] | None:
    model_id = str(item.get("id") or "").strip()
    if not model_id:
        return None
    name = str(item.get("name") or "").strip() or model_id_to_name(model_id)
    family = ""
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        family = str(capabilities.get("family") or "").strip()
    suffix = f" ({family})" if family else ""
    return {
        "id": model_id,
        "name": name,
        "description": f"{name} via Copilot{suffix}",
    }


async def fetch_copilot_api_models(
    auth: CopilotAuth | None,
    *,
    timeout: float = 30.0,
) -> list[dict[str, str]]:
    if not auth or not auth.access_token or auth.is_expired():
        return []

    api_base = _copilot_api_base(getattr(auth, "enterprise_url", None))
    headers = {
        "Authorization": f"Bearer {auth.access_token}",
        "Accept": "application/json",
        "User-Agent": "co-scientist/copilot",
        "Editor-Version": "vscode/1.99.0",
        "Editor-Plugin-Version": "copilot-chat/0.26.7",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(f"{api_base}/models", headers=headers)
    if not response.is_success:
        response.raise_for_status()
    payload = response.json()
    raw_models = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(raw_models, list):
        return []

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict) or not _copilot_model_enabled(item):
            continue
        model = _format_copilot_api_model(item)
        if not model or model["id"] in seen:
            continue
        seen.add(model["id"])
        results.append(model)
    return results


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
