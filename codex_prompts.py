"""Codex CLI system instructions fetching.

The ChatGPT Codex backend validates the `instructions` field.
The official Codex CLI fetches model-family prompt markdown from the
`openai/codex` GitHub repository (latest release tag) and caches it locally.

This module mirrors that behavior (in a lightweight Python form) to keep
requests compatible with the backend.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx


GITHUB_API_RELEASES = "https://api.github.com/repos/openai/codex/releases/latest"
GITHUB_HTML_RELEASES = "https://github.com/openai/codex/releases/latest"

ModelFamily = Literal[
    "gpt-5.2-codex",
    "codex-max",
    "codex",
    "gpt-5.2",
    "gpt-5.1",
]

PROMPT_FILES: dict[ModelFamily, str] = {
    "gpt-5.2-codex": "gpt-5.2-codex_prompt.md",
    "codex-max": "gpt-5.1-codex-max_prompt.md",
    "codex": "gpt_5_codex_prompt.md",
    "gpt-5.2": "gpt_5_2_prompt.md",
    "gpt-5.1": "gpt_5_1_prompt.md",
}

CACHE_FILES: dict[ModelFamily, str] = {
    "gpt-5.2-codex": "gpt-5.2-codex-instructions.md",
    "codex-max": "codex-max-instructions.md",
    "codex": "codex-instructions.md",
    "gpt-5.2": "gpt-5.2-instructions.md",
    "gpt-5.1": "gpt-5.1-instructions.md",
}


@dataclass
class _CacheMetadata:
    etag: str | None
    tag: str | None
    last_checked: float | None
    url: str | None


def _cache_dir() -> Path:
    # Prefer a per-user cache directory.
    # Windows: %APPDATA%\langchain_antigravity\cache
    # Others:  ~/.cache/langchain_antigravity
    appdata = os.getenv("APPDATA")
    if appdata:
        base = Path(appdata) / "langchain_antigravity" / "cache"
    else:
        base = Path.home() / ".cache" / "langchain_antigravity"
    return base / "codex"


def get_model_family(normalized_model: str) -> ModelFamily:
    m = (normalized_model or "").lower()

    # Order matters: more specific first.
    if "gpt-5.2-codex" in m or "gpt 5.2 codex" in m:
        return "gpt-5.2-codex"
    if "codex-max" in m:
        return "codex-max"
    if "codex" in m or m.startswith("codex-"):
        return "codex"
    if "gpt-5.2" in m:
        return "gpt-5.2"
    return "gpt-5.1"


def _read_metadata(meta_path: Path) -> _CacheMetadata:
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return _CacheMetadata(
            etag=data.get("etag"),
            tag=data.get("tag"),
            last_checked=data.get("lastChecked"),
            url=data.get("url"),
        )
    except Exception:
        return _CacheMetadata(etag=None, tag=None, last_checked=None, url=None)


def _write_metadata(meta_path: Path, meta: _CacheMetadata) -> None:
    meta_path.write_text(
        json.dumps(
            {
                "etag": meta.etag,
                "tag": meta.tag,
                "lastChecked": meta.last_checked,
                "url": meta.url,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _get_latest_release_tag(client: httpx.Client) -> str:
    # Prefer the GitHub API (fast, stable).
    try:
        resp = client.get(GITHUB_API_RELEASES, headers={"Accept": "application/vnd.github+json"})
        if resp.status_code == 200:
            data = resp.json()
            tag = data.get("tag_name")
            if isinstance(tag, str) and tag:
                return tag
    except Exception:
        pass

    # Fallback to HTML redirect parsing.
    resp = client.get(GITHUB_HTML_RELEASES, follow_redirects=True)
    resp.raise_for_status()

    # If redirected to /tag/<tag>
    final_url = str(resp.url)
    if "/tag/" in final_url:
        tag = final_url.split("/tag/")[-1]
        if tag and "/" not in tag:
            return tag

    html = resp.text
    match = re.search(r"/openai/codex/releases/tag/([^\"/]+)", html)
    if match:
        return match.group(1)

    raise RuntimeError("Failed to determine latest release tag from GitHub")


def get_codex_instructions(normalized_model: str = "gpt-5.1-codex") -> str:
    """Return the official Codex CLI instructions for this model family.

    Uses a disk cache with ETag and a 15-minute TTL.

    To bypass GitHub fetching (e.g., on restricted networks), set the
    LANGCHAIN_ANTIGRAVITY_CODEX_INSTRUCTIONS environment variable to the
    official Codex prompt content. If set, this function will return that
    value directly without any network calls or caching.
    """

    # Optional override for power-users.
    env_override = os.getenv("LANGCHAIN_ANTIGRAVITY_CODEX_INSTRUCTIONS")
    if env_override:
        print(f"[langchain_antigravity] Using LANGCHAIN_ANTIGRAVITY_CODEX_INSTRUCTIONS override for {normalized_model}")
        return env_override

    model_family = get_model_family(normalized_model)
    prompt_file = PROMPT_FILES[model_family]

    cache_dir = _cache_dir()
    cache_file = cache_dir / CACHE_FILES[model_family]
    meta_file = cache_dir / (CACHE_FILES[model_family].replace(".md", "-meta.json"))

    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = _read_metadata(meta_file) if meta_file.exists() else _CacheMetadata(None, None, None, None)

    # Rate limit protection: if checked within 15 minutes and we have a file, use it.
    ttl_seconds = 15 * 60
    if meta.last_checked and (time.time() - float(meta.last_checked)) < ttl_seconds and cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            latest_tag = _get_latest_release_tag(client)
            url = f"https://raw.githubusercontent.com/openai/codex/{latest_tag}/codex-rs/core/{prompt_file}"

            headers: dict[str, str] = {}
            if meta.tag != latest_tag:
                meta.etag = None
            if meta.etag:
                headers["If-None-Match"] = meta.etag

            resp = client.get(url, headers=headers)

            if resp.status_code == 304 and cache_file.exists():
                meta.tag = latest_tag
                meta.last_checked = time.time()
                meta.url = url
                _write_metadata(meta_file, meta)
                return cache_file.read_text(encoding="utf-8")

            resp.raise_for_status()

            instructions = resp.text
            cache_file.write_text(instructions, encoding="utf-8")

            meta.etag = resp.headers.get("etag")
            meta.tag = latest_tag
            meta.last_checked = time.time()
            meta.url = url
            _write_metadata(meta_file, meta)

            return instructions

    except Exception as exc:
        # If GitHub is blocked/unavailable, try stale cache.
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        raise RuntimeError(
            "Failed to fetch Codex instructions from GitHub, and no cache is available. "
            "If you are on a restricted network, either allow access to github.com/raw.githubusercontent.com "
            "or set LANGCHAIN_ANTIGRAVITY_CODEX_INSTRUCTIONS to the official Codex prompt content."
        ) from exc
