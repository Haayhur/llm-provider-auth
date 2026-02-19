"""
LangChain ChatModel for OpenAI Codex API.

This module provides a LangChain BaseChatModel that communicates with OpenAI's
ChatGPT Codex backend via OAuth authentication, allowing access to GPT-5.x
and Codex models using a ChatGPT Plus/Pro subscription.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Callable, Sequence

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from .usage_helpers import _extract_usage_metadata, _normalize_token_usage

try:
    from .codex_auth import (
        CodexAuth,
        extract_account_id,
        load_codex_auth_from_storage,
        refresh_codex_token,
        normalize_codex_model,
    )
    from .constants import (
        CODEX_BASE_URL,
        CODEX_DUMMY_API_KEY,
        CODEX_HEADERS,
        CODEX_MODEL_MAPPINGS,
    )
    from .codex_prompts import get_codex_instructions
except ImportError:  # pragma: no cover
    from codex_auth import (  # type: ignore
        CodexAuth,
        extract_account_id,
        load_codex_auth_from_storage,
        refresh_codex_token,
        normalize_codex_model,
    )
    from constants import (  # type: ignore
        CODEX_BASE_URL,
        CODEX_DUMMY_API_KEY,
        CODEX_HEADERS,
        CODEX_MODEL_MAPPINGS,
    )
    from codex_prompts import get_codex_instructions  # type: ignore


_LOGGER = logging.getLogger(__name__)
_AUTH_METRICS: dict[str, int] = {
    "codex_unauthorized_retry_attempt": 0,
    "codex_unauthorized_retry_success": 0,
    "codex_unauthorized_retry_fail": 0,
}


def _record_auth_metric(metric: str, **context: str | None) -> None:
    if metric not in _AUTH_METRICS:
        _AUTH_METRICS[metric] = 0
    _AUTH_METRICS[metric] += 1
    context_str = ", ".join(
        f"{k}={v}" for k, v in context.items()
        if v is not None and str(v).strip()
    )
    suffix = f" ({context_str})" if context_str else ""
    _LOGGER.info(
        "metric=%s count=%d%s",
        metric,
        _AUTH_METRICS[metric],
        suffix,
    )


class ChatCodex(BaseChatModel):
    """
    LangChain ChatModel for OpenAI Codex API via ChatGPT OAuth.

    Supports:
    - GPT-5.2 models (none/low/medium/high/xhigh)
    - GPT-5.2 Codex (low/medium/high/xhigh)
    - GPT-5.1 Codex Max (low/medium/high/xhigh)
    - GPT-5.1 Codex (low/medium/high)
    - GPT-5.1 Codex Mini (medium/high)
    - GPT-5.1 (none/low/medium/high)

    Example:
        ```python
        from langchain_antigravity import ChatCodex

        chat = ChatCodex(model="gpt-5.2-codex")
        response = chat.invoke("Hello!")
        ```
    """

    model: str = Field(default="gpt-5.2-codex")
    """Model name to use."""

    temperature: float | None = Field(default=None)
    """Sampling temperature."""

    max_tokens: int | None = Field(default=None)
    """Maximum output tokens."""

    reasoning_effort: str | None = Field(default=None)
    """Reasoning effort: none, low, medium, high, xhigh."""

    auth: CodexAuth | None = Field(default=None, exclude=True)
    """Authentication state. If not provided, will load from storage."""

    account_id: str | None = Field(default=None)
    """Optional ChatGPT account ID override."""

    on_auth_update: Callable[[CodexAuth], Any] | None = Field(default=None, exclude=True)
    """Optional callback invoked after auth refresh to persist updated tokens."""

    _tools: list[dict[str, Any]] = []
    _auth_refresh_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @property
    def _llm_type(self) -> str:
        return "codex"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def bind_tools(
        self,
        tools: "Sequence[dict[str, Any] | type | Callable | BaseTool]",
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ChatCodex":
        """Bind tools to model."""
        function_declarations = []

        for tool in tools:
            if isinstance(tool, dict):
                # Accept either an already-shaped Responses tool object or a plain
                # function schema with name/description/parameters.
                if tool.get("type") == "function" and isinstance(tool.get("name"), str):
                    function_declarations.append(tool)
                else:
                    function_declarations.append({
                        "type": "function",
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}) or {},
                    })
            elif isinstance(tool, type):
                schema = {}
                function_declarations.append({
                    "type": "function",
                    "name": tool.__name__,
                    "description": tool.__doc__ or "",
                    "parameters": schema,
                })
            elif isinstance(tool, Callable) and not isinstance(tool, type):
                schema = {}
                function_declarations.append({
                    "type": "function",
                    "name": tool.__name__,
                    "description": tool.__doc__ or "",
                    "parameters": schema,
                })
            elif hasattr(tool, "name") and hasattr(tool, "description"):
                schema: Any = {}
                try:
                    if hasattr(tool, "args_schema") and tool.args_schema:
                        schema_result = getattr(tool.args_schema, "model_json_schema", lambda: {})()
                        if isinstance(schema_result, dict):
                            schema = schema_result
                except Exception:
                    pass

                function_declarations.append({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                })

        new_model = self.model_copy()
        new_model._tools = function_declarations
        return new_model

    async def _ensure_auth(self) -> CodexAuth:
        """Ensure we have valid authentication."""
        if self.auth is None:
            self.auth = load_codex_auth_from_storage()

        if self.auth is None:
            raise ValueError(
                "No Codex authentication found. "
                "Run 'codex-auth login' first or provide auth parameter."
            )

        if self.auth.is_expired():
            async with self._auth_refresh_lock:
                if self.auth and self.auth.is_expired():
                    self.auth = await refresh_codex_token(self.auth)
                    if self.auth:
                        self._persist_auth_update(self.auth)

        # Ensure we have an account_id for the required chatgpt header.
        # (Older stored accounts may be missing this field.)
        if not self.auth.account_id and self.auth.access_token:
            self.auth.account_id = extract_account_id(self.auth.access_token)
            self._persist_auth_update(self.auth)

        return self.auth

    def _persist_auth_update(self, auth: CodexAuth) -> None:
        callback = self.on_auth_update
        if not callable(callback):
            return
        try:
            callback(auth)
        except Exception:
            _LOGGER.warning("Failed to persist refreshed Codex auth", exc_info=True)

    def _is_cloudflare_challenge(self, status_code: int, text: str) -> bool:
        if status_code in (403, 429) and "cdn-cgi/challenge" in text:
            return True
        if status_code in (403, 429) and "Enable JavaScript and cookies" in text:
            return True
        return False

    def _extract_error_message(self, response: httpx.Response) -> str:
        error_text = response.text
        try:
            error_data = response.json()
            if "error" in error_data:
                error_obj = error_data["error"]
                if isinstance(error_obj, dict):
                    return str(error_obj.get("message", error_text))
                return str(error_obj)
        except (json.JSONDecodeError, KeyError):
            pass
        return error_text

    async def _recover_after_unauthorized(self, auth: CodexAuth) -> bool:
        """Try to recover once when the backend rejects an existing access token."""
        _record_auth_metric(
            "codex_unauthorized_retry_attempt",
            account_id=auth.account_id,
            email=auth.email,
        )
        async with self._auth_refresh_lock:
            latest_auth = self.auth
            if latest_auth and latest_auth.access_token and latest_auth.access_token != auth.access_token:
                _record_auth_metric(
                    "codex_unauthorized_retry_success",
                    account_id=latest_auth.account_id,
                    email=latest_auth.email,
                )
                return True

            latest = load_codex_auth_from_storage()
            if latest and latest.refresh_token:
                same_account = (
                    not auth.account_id
                    or not latest.account_id
                    or auth.account_id == latest.account_id
                )
                if same_account and latest.refresh_token != auth.refresh_token:
                    latest = await refresh_codex_token(latest)
                    self.auth = latest
                    self._persist_auth_update(latest)
                    _record_auth_metric(
                        "codex_unauthorized_retry_success",
                        account_id=latest.account_id,
                        email=latest.email,
                    )
                    return True

            try:
                refreshed = await refresh_codex_token(auth)
            except Exception:
                _record_auth_metric(
                    "codex_unauthorized_retry_fail",
                    account_id=auth.account_id,
                    email=auth.email,
                )
                raise

            self.auth = refreshed
            self._persist_auth_update(refreshed)
            _record_auth_metric(
                "codex_unauthorized_retry_success",
                account_id=refreshed.account_id,
                email=refreshed.email,
            )
            return True

    def _parse_sse_final_response(self, sse_text: str) -> dict[str, Any] | None:
        """Extract the final Responses payload from an SSE stream."""
        for raw_line in sse_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                continue
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            evt_type = evt.get("type")
            if evt_type in ("response.done", "response.completed"):
                response_obj = evt.get("response")
                if isinstance(response_obj, dict):
                    return response_obj
        return None

    def _parse_responses_api_output(self, response_data: dict[str, Any]) -> AIMessage:
        """Parse Responses-style payload into an AIMessage."""
        output = response_data.get("output", [])
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "message":
                    # Responses API: message.content is usually an array of content blocks.
                    content = item.get("content")
                    if isinstance(content, str):
                        if content:
                            content_parts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            block_type = block.get("type")
                            if block_type in ("output_text", "text"):
                                text = block.get("text") or block.get("value")
                                if isinstance(text, str) and text:
                                    content_parts.append(text)
                elif item_type in ("function_call", "tool_call"):
                    name = item.get("name") or ""
                    args_raw = item.get("arguments")
                    args: Any = {}
                    if isinstance(args_raw, dict):
                        args = args_raw
                    elif isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw) if args_raw else {}
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append(
                        ToolCall(name=name, args=args, id=item.get("id", ""))
                    )

        return AIMessage(content="".join(content_parts), tool_calls=tool_calls)

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert LangChain messages to ChatGPT Codex backend `input` format."""
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                input_items.append({
                    "type": "message",
                    "role": "developer",
                    "content": msg.content,
                })
            elif isinstance(msg, HumanMessage):
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": msg.content,
                })
            elif isinstance(msg, AIMessage):
                # Preserve assistant text in history when provided.
                content = msg.content or ""
                if content:
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    })
            elif isinstance(msg, ToolMessage):
                # Tool results can be represented as messages for minimal compatibility.
                # (Full function_call_output wiring can be added later if needed.)
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": msg.content,
                })

        return input_items

    def _build_request_body(
        self,
        input_items: list[dict[str, Any]],
        instructions: str,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build Codex API request body."""
        effective_model = normalize_codex_model(self.model)

        # ChatGPT Codex backend requirements (mirrors Codex CLI behavior):
        # - `instructions` is required
        # - `store=false` is required
        # - It expects `input` items, not `messages`
        # - It streams via SSE; we parse SSE even for non-stream calls.
        body: dict[str, Any] = {
            "model": effective_model,
            "instructions": instructions,
            "store": False,
            "stream": True,
            "input": input_items,
        }

        if self.temperature is not None:
            body["temperature"] = self.temperature

        if self.max_tokens is not None:
            body["max_output_tokens"] = self.max_tokens

        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}

        if self._tools:
            body["tools"] = self._tools

        return body

    def _parse_response(self, response_data: dict[str, Any]) -> AIMessage:
        """Parse Codex API response into AIMessage.

        Supports both:
        - Chat Completions-like shape (choices[0].message)
        - Responses API shape (output[])
        """

        if "output" in response_data:
            base_message = self._parse_responses_api_output(response_data)
            usage_metadata, raw_usage = _extract_usage_metadata(response_data)
            response_metadata = dict(getattr(base_message, "response_metadata", {}) or {})
            if raw_usage:
                response_metadata["usage"] = raw_usage
            if hasattr(base_message, "model_copy"):
                return base_message.model_copy(
                    update={
                        "usage_metadata": usage_metadata or None,
                        "response_metadata": response_metadata or None,
                    }
                )
            return AIMessage(
                content=base_message.content,
                tool_calls=base_message.tool_calls,
                usage_metadata=usage_metadata or None,
                response_metadata=response_metadata or None,
            )

        choices = response_data.get("choices", [])
        if not choices:
            return AIMessage(content="")

        choice = choices[0]
        message = choice.get("message", {})

        content = message.get("content", "")
        tool_calls: list[ToolCall] = []

        if "tool_calls" in message and isinstance(message.get("tool_calls"), list):
            for tc in message["tool_calls"]:
                function = tc.get("function", {}) if isinstance(tc, dict) else {}
                args_str = function.get("arguments", "{}") if isinstance(function, dict) else "{}"
                try:
                    args_obj = json.loads(args_str) if isinstance(args_str, str) else {}
                except json.JSONDecodeError:
                    args_obj = {}
                tool_calls.append(
                    ToolCall(
                        name=function.get("name", "") if isinstance(function, dict) else "",
                        args=args_obj,
                        id=tc.get("id", "") if isinstance(tc, dict) else "",
                    )
                )

        usage_metadata, raw_usage = _extract_usage_metadata(response_data)
        response_metadata = {"usage": raw_usage} if raw_usage else {}
        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata or None,
            response_metadata=response_metadata or None,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response asynchronously."""
        auth = await self._ensure_auth()
        input_items = self._convert_messages(messages)
        effective_model = normalize_codex_model(self.model)
        instructions = get_codex_instructions(effective_model)
        body = self._build_request_body(input_items, instructions, stream=False)

        headers = {
            **CODEX_HEADERS,
            "Authorization": f"Bearer {auth.access_token}",
            "Content-Type": "application/json",
            # Codex backend returns SSE even for non-stream; match Codex CLI behavior.
            "Accept": "text/event-stream",
        }

        if auth.account_id:
            headers["chatgpt-account-id"] = auth.account_id

        # IMPORTANT: Codex uses /codex/responses (not /responses).
        url = f"{CODEX_BASE_URL}/codex/responses"

        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(2):
                response = await client.post(url, headers=headers, json=body)

                if not response.is_success:
                    error_text = self._extract_error_message(response)
                    if self._is_cloudflare_challenge(response.status_code, response.text):
                        raise RuntimeError(
                            "Codex request was blocked by Cloudflare on chatgpt.com (JS/cookie challenge). "
                            "This usually happens when hitting the non-Codex endpoint or when requests are missing "
                            "required headers. Re-run 'codex-auth login', ensure you are using the latest package, "
                            "and retry."
                        )
                    if response.status_code == 401 and attempt == 0:
                        await self._recover_after_unauthorized(auth)
                        auth = self.auth or auth
                        headers["Authorization"] = f"Bearer {auth.access_token}"
                        if auth.account_id:
                            headers["chatgpt-account-id"] = auth.account_id
                        continue
                    raise RuntimeError(f"Codex API error: {response.status_code} - {error_text}")

                # Codex backend often returns SSE even when stream=false.
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type.lower() or response.text.lstrip().startswith("data: "):
                    final_response = self._parse_sse_final_response(response.text)
                    if not final_response:
                        raise RuntimeError("Codex API returned SSE but no final response.done event was found")
                    message = self._parse_response(final_response)
                else:
                    try:
                        response_data = response.json()
                        message = self._parse_response(response_data)
                    except json.JSONDecodeError:
                        # Some deployments still return SSE without a helpful content-type.
                        final_response = self._parse_sse_final_response(response.text)
                        if not final_response:
                            raise
                        message = self._parse_response(final_response)

                llm_output = {"model_name": self.model}
                if message.usage_metadata:
                    llm_output["token_usage"] = _normalize_token_usage(message.usage_metadata)
                return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)

            raise RuntimeError("Codex API error: unauthorized after token refresh")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response synchronously."""
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop, None, **kwargs)
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream a response asynchronously."""
        auth = await self._ensure_auth()
        input_items = self._convert_messages(messages)
        effective_model = normalize_codex_model(self.model)
        instructions = get_codex_instructions(effective_model)
        body = self._build_request_body(input_items, instructions, stream=True)

        headers = {
            **CODEX_HEADERS,
            "Authorization": f"Bearer {auth.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        if auth.account_id:
            headers["chatgpt-account-id"] = auth.account_id

        url = f"{CODEX_BASE_URL}/codex/responses"

        async with httpx.AsyncClient(timeout=120.0) as client:
            unauthorized_retry_attempted = False
            saw_any_delta = False
            for attempt in range(2):
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    if not response.is_success:
                        error_bytes = await response.aread()
                        error_text = error_bytes.decode("utf-8", errors="replace")
                        if self._is_cloudflare_challenge(response.status_code, error_text):
                            raise RuntimeError(
                                "Codex streaming request was blocked by Cloudflare on chatgpt.com (JS/cookie challenge). "
                                "Re-run 'codex-auth login' and retry."
                            )
                        if response.status_code == 401 and attempt == 0:
                            unauthorized_retry_attempted = True
                            await self._recover_after_unauthorized(auth)
                            auth = self.auth or auth
                            headers["Authorization"] = f"Bearer {auth.access_token}"
                            if auth.account_id:
                                headers["chatgpt-account-id"] = auth.account_id
                            continue
                        try:
                            error_data = json.loads(error_text)
                            if "error" in error_data:
                                error_obj = error_data["error"]
                                if isinstance(error_obj, dict):
                                    error_text = error_obj.get("message", error_text)
                                else:
                                    error_text = str(error_obj)
                        except (json.JSONDecodeError, KeyError):
                            pass
                        raise RuntimeError(f"Codex API error: {response.status_code} - {error_text}")

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                return

                            try:
                                chunk_data = json.loads(data)
                                evt_type = chunk_data.get("type")

                                # Responses API streaming text
                                if evt_type in ("response.output_text.delta", "response.output_text.fragment"):
                                    delta_text = chunk_data.get("delta") or chunk_data.get("text")
                                    if isinstance(delta_text, str) and delta_text:
                                        saw_any_delta = True
                                        yield ChatGenerationChunk(message=AIMessageChunk(content=delta_text))
                                        continue

                                # Some backends emit message deltas under nested paths.
                                if evt_type == "response.output_item.delta":
                                    delta = chunk_data.get("delta")
                                    if isinstance(delta, dict):
                                        content = delta.get("content")
                                        if isinstance(content, list):
                                            for block in content:
                                                if not isinstance(block, dict):
                                                    continue
                                                if block.get("type") in ("output_text", "text"):
                                                    text = block.get("text") or block.get("value")
                                                    if isinstance(text, str) and text:
                                                        saw_any_delta = True
                                                        yield ChatGenerationChunk(
                                                            message=AIMessageChunk(content=text)
                                                        )
                            except json.JSONDecodeError:
                                continue
                    if saw_any_delta:
                        _LOGGER.warning("Codex stream ended without [DONE]; returning partial response")
                        return
                    raise RuntimeError("Codex API stream ended unexpectedly before completion event")
            if unauthorized_retry_attempted:
                raise RuntimeError("Codex API error: unauthorized after token refresh")
            raise RuntimeError("Codex API stream ended unexpectedly before completion event")
