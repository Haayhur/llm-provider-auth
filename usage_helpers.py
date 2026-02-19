"""Shared usage parsing helpers for chat models."""

from __future__ import annotations

from typing import Any


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_usage_metadata(response_data: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    inner = response_data.get("response", response_data)
    usage = inner.get("usageMetadata") or inner.get("usage") or inner.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        return {}, {}

    prompt = _coerce_int(
        usage.get("promptTokenCount")
        or usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("promptTokens")
        or usage.get("prompt_token_count")
    )
    completion = _coerce_int(
        usage.get("candidatesTokenCount")
        or usage.get("candidates_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("candidate_tokens")
    )
    total = _coerce_int(
        usage.get("totalTokenCount")
        or usage.get("total_tokens")
        or usage.get("totalTokenCount")
    )

    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    if prompt is None and completion is None and total is None:
        return {}, usage

    prompt_tokens = prompt or 0
    completion_tokens = completion or 0
    total_tokens = total if total is not None else prompt_tokens + completion_tokens

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }, usage


def _normalize_token_usage(usage_meta: dict[str, int]) -> dict[str, int]:
    prompt = usage_meta.get("prompt_tokens") or usage_meta.get("input_tokens") or 0
    completion = usage_meta.get("completion_tokens") or usage_meta.get("output_tokens") or 0
    total = usage_meta.get("total_tokens") or (prompt + completion)
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
    }
