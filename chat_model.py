"""
LangChain ChatModel for Antigravity API.

This module provides a LangChain BaseChatModel that communicates with Google's
Antigravity API, allowing access to Gemini 3 and Claude models.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from typing import Any, AsyncIterator, Literal

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
 
# Local imports (support both editable installs and vendored copies)
try:
    from . import auth as antigravity_auth
    from .auth import AntigravityAuth
    from . import constants, schema
except Exception:  # pragma: no cover
    import auth as antigravity_auth  # type: ignore
    from auth import AntigravityAuth  # type: ignore
    import constants  # type: ignore
    import schema  # type: ignore

from .chat_model_helpers import (
    _env_project_override,
    _resolve_auth_project_id,
    _retry_after_seconds,
    get_header_style,
    get_thinking_config,
    is_claude_model,
    is_thinking_model,
    resolve_model_name,
)
from .usage_helpers import _extract_usage_metadata, _normalize_token_usage
from .chat_message_utils import convert_messages


# Basic in-process concurrency limiter to avoid hammering Antigravity endpoints.
# This is especially important for websocket streaming in the main app.
_DEFAULT_MAX_CONCURRENCY = 2
try:
    _MAX_CONCURRENCY = int(os.environ.get("ANTIGRAVITY_MAX_CONCURRENCY", str(_DEFAULT_MAX_CONCURRENCY)))
except Exception:
    _MAX_CONCURRENCY = _DEFAULT_MAX_CONCURRENCY
_MAX_CONCURRENCY = max(1, _MAX_CONCURRENCY)
_ANTIGRAVITY_HTTP_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENCY)




async def _read_response_text(response: httpx.Response) -> str:
    try:
        return response.text or ""
    except httpx.ResponseNotRead:
        try:
            body = await response.aread()
        except Exception:
            return ""
        encoding = response.encoding or "utf-8"
        try:
            return body.decode(encoding, errors="replace")
        except Exception:
            return body.decode("utf-8", errors="replace")
    except Exception:
        return ""


class ChatAntigravity(BaseChatModel):
    """
    LangChain ChatModel for Antigravity API.
    
    Supports:
    - Gemini 3 models (Flash, Pro Low, Pro High)
    - Claude models (Sonnet 4.5, Opus 4.5)
    - Extended thinking for supported models
    - Tool/function calling
    
    Example:
        ```python
        from langchain_antigravity import ChatAntigravity
        
        chat = ChatAntigravity(model="antigravity-claude-sonnet-4-5")
        response = chat.invoke("Hello!")
        ```
    """
    
    model: str = Field(default="antigravity-gemini-3-flash")
    """Model name to use."""
    
    temperature: float | None = Field(default=None)
    """Sampling temperature."""
    
    max_output_tokens: int | None = Field(default=None)
    """Maximum output tokens."""
    
    auth: AntigravityAuth | None = Field(default=None, exclude=True)
    """Authentication state. If not provided, will load from storage."""
    
    project_id: str | None = Field(default=None)
    """Optional project ID override (takes precedence over env/auth)."""
    
    _tools: list[dict[str, Any]] = []
    """Bound tools."""
    
    @property
    def _llm_type(self) -> str:
        return "antigravity"
    
    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}
    
    def bind_tools(self, tools: list[Any]) -> "ChatAntigravity":
        """Bind tools to the model."""
        function_declarations = []
        
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                # LangChain tool
                tool_schema: dict[str, Any] = {}
                if hasattr(tool, "args_schema") and tool.args_schema:
                    tool_schema = tool.args_schema.model_json_schema()
                    tool_schema = schema.clean_json_schema_for_antigravity(tool_schema)
                
                # Add parameter signature to description
                description = tool.description
                if tool_schema.get("properties"):
                    sig = schema.format_parameter_signature(
                        tool_schema["properties"],
                        tool_schema.get("required", []),
                    )
                    if sig:
                        description = f"{description}\n\nSTRICT PARAMETERS: {sig}."
                
                function_declarations.append({
                    "name": tool.name,
                    "description": description,
                    "parameters": tool_schema,
                })
            elif isinstance(tool, dict):
                # Raw dict tool definition
                function_declarations.append(tool)
        
        new_model = self.model_copy()
        new_model._tools = function_declarations
        return new_model
    
    async def _ensure_auth(self) -> AntigravityAuth:
        """Ensure we have valid authentication."""
        if self.auth is None:
            self.auth = antigravity_auth.load_auth_from_storage()
        
        if self.auth is None:
            raise ValueError(
                "No Antigravity authentication found. "
                "Run 'opencode auth login' first or provide auth parameter."
            )
        
        if self.auth.is_expired():
            self.auth = await antigravity_auth.refresh_access_token(self.auth)
        
        return self.auth
    
    def _build_request_body(
        self,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the Antigravity request body."""
        effective_model = resolve_model_name(self.model)
        project_override = self.project_id or _env_project_override()
        project_id = project_override or _resolve_auth_project_id(self.auth)
        
        # Build generation config
        generation_config: dict[str, Any] = {}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = self.max_output_tokens
        
        # Add thinking config if applicable
        thinking_config = get_thinking_config(self.model)
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config
        
        # Build request
        request: dict[str, Any] = {
            "contents": contents,
            "sessionId": f"session-{secrets.token_hex(16)}",
        }

        if generation_config:
            request["generationConfig"] = generation_config

        if system_instruction:
            # Add Claude tool hardening if tools are present
            if self._tools and is_claude_model(self.model):
                system_text = system_instruction["parts"][0].get("text", "")
                system_instruction = {
                    "parts": [{"text": f"{system_text}\n\n{constants.CLAUDE_TOOL_SYSTEM_INSTRUCTION}"}]
                }
            request["systemInstruction"] = system_instruction
        elif self._tools and is_claude_model(self.model):
            # Add tool hardening as system instruction if no system message
            request["systemInstruction"] = {
                "parts": [{"text": constants.CLAUDE_TOOL_SYSTEM_INSTRUCTION}]
            }

        if self._tools:
            request["tools"] = [{"functionDeclarations": self._tools}]
            # Claude requires VALIDATED mode for tool calling
            if is_claude_model(self.model):
                request["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "VALIDATED",
                    }
                }

        # Wrap in Antigravity envelope
        envelope = {
            "model": effective_model,
            "request": request,
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": f"agent-{secrets.token_hex(16)}",
        }
        if project_id:
            envelope["project"] = project_id
        return envelope
    
    def _parse_response(self, response_data: dict[str, Any]) -> AIMessage:
        """Parse Antigravity response into AIMessage."""
        # Unwrap response envelope
        inner = response_data.get("response", response_data)
        
        candidates = inner.get("candidates", [])
        if not candidates:
            return AIMessage(content="")
        
        candidate = candidates[0]
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])
        
        text_parts = []
        tool_calls = []
        
        for part in parts:
            if "text" in part and not part.get("thought"):
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        name=fc.get("name", ""),
                        args=fc.get("args", {}),
                        id=fc.get("id", ""),
                    )
                )
        
        content = "".join(text_parts)

        usage_metadata, raw_usage = _extract_usage_metadata(response_data)
        response_metadata = {"usage": raw_usage} if raw_usage else {}

        return AIMessage(
            content=content,
            tool_calls=tool_calls if tool_calls else [],
            usage_metadata=usage_metadata or None,
            response_metadata=response_metadata or None,
        )
    
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response asynchronously."""
        last_http_error: httpx.HTTPStatusError | None = None
        last_status_by_endpoint: dict[str, int] = {}

        for attempt in range(4):
            auth = await self._ensure_auth()
            contents, system_instruction = convert_messages(messages)
            body = self._build_request_body(contents, system_instruction)

            header_style = get_header_style(self.model)
            headers = constants.ANTIGRAVITY_HEADERS if header_style == "antigravity" else constants.GEMINI_CLI_HEADERS
            headers = {
                **headers,
                "Authorization": f"Bearer {auth.access_token}",
                "Content-Type": "application/json",
            }

            if is_thinking_model(self.model) and is_claude_model(self.model):
                headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

            async with httpx.AsyncClient(timeout=120.0) as client:
                for endpoint in constants.ANTIGRAVITY_ENDPOINT_FALLBACKS:
                    url = f"{endpoint}/v1internal:generateContent"

                    try:
                        async with _ANTIGRAVITY_HTTP_SEMAPHORE:
                            response = await client.post(url, headers=headers, json=body)
                        last_status_by_endpoint[endpoint] = response.status_code

                        if response.status_code == 429 or response.status_code >= 500:
                            # Honor server guidance if available; otherwise backoff.
                            base = min(8.0, 0.5 * (2**attempt))
                            await asyncio.sleep(_retry_after_seconds(response, default=base))
                            continue

                        # Sandbox endpoints may return 403 while prod works.
                        if response.status_code in (401, 403, 404):
                            try:
                                response.raise_for_status()
                            except httpx.HTTPStatusError as e:
                                last_http_error = e
                            continue

                        response.raise_for_status()

                        response_data = response.json()
                        message = self._parse_response(response_data)
                        llm_output = {"model_name": self.model}
                        if message.usage_metadata:
                            llm_output["token_usage"] = _normalize_token_usage(message.usage_metadata)
                        return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)

                    except httpx.HTTPStatusError as e:
                        last_http_error = e
                        if e.response.status_code in (429, 500, 502, 503, 504):
                            base = min(8.0, 0.5 * (2**attempt))
                            await asyncio.sleep(_retry_after_seconds(e.response, default=base))
                            continue
                        continue
                    except httpx.RequestError:
                        continue

            if attempt == 0:
                did_retry = False
                if last_http_error is not None and last_http_error.response.status_code in (401, 403):
                    try:
                        if self.auth is not None:
                            self.auth = await antigravity_auth.refresh_access_token(self.auth)
                            did_retry = True
                    except Exception:
                        pass

                project_override = self.project_id or _env_project_override()
                auth_project = _resolve_auth_project_id(self.auth)
                needs_project_refresh = (
                    not did_retry
                    and not project_override
                    and self.auth is not None
                    and (not auth_project or auth_project == constants.ANTIGRAVITY_DEFAULT_PROJECT_ID)
                    and any(status in (403, 429) for status in last_status_by_endpoint.values())
                )
                if needs_project_refresh:
                    try:
                        project_id = await antigravity_auth.fetch_project_id(self.auth.access_token)
                        if project_id:
                            self.auth.project_id = project_id
                            did_retry = True
                    except Exception:
                        pass

                if did_retry:
                    continue

            # If we were primarily rate-limited, wait a bit and retry.
            if any(status == 429 for status in last_status_by_endpoint.values()):
                await asyncio.sleep(min(10.0, 1.0 * (2**attempt)))
                continue

        details = ", ".join(f"{k}={v}" for k, v in last_status_by_endpoint.items())
        if last_http_error is not None:
            # If any endpoint rate-limited us, surface 429 even if the last HTTP error was a sandbox 403.
            code = 429 if any(status == 429 for status in last_status_by_endpoint.values()) else last_http_error.response.status_code
            body_snippet = (await _read_response_text(last_http_error.response)).strip()
            if len(body_snippet) > 500:
                body_snippet = body_snippet[:500] + "…"
            raise RuntimeError(
                "Antigravity request failed for all endpoints. "
                f"Last status={code}. Endpoint statuses: {details}. "
                "If you see 401/403, re-authenticate via the Antigravity OAuth flow and ensure your account has access. "
                f"Response: {body_snippet}"
            ) from last_http_error

        raise RuntimeError(f"All Antigravity endpoints failed. Endpoint statuses: {details}")
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response synchronously."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop, run_manager, **kwargs)
        )
    
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream a response asynchronously."""
        last_http_error: httpx.HTTPStatusError | None = None
        last_status_by_endpoint: dict[str, int] = {}

        # Streaming is commonly used in the main app; allow a few retries with backoff for 429/5xx.
        for attempt in range(4):
            auth = await self._ensure_auth()
            contents, system_instruction = convert_messages(messages)
            body = self._build_request_body(contents, system_instruction)

            header_style = get_header_style(self.model)
            headers = constants.ANTIGRAVITY_HEADERS if header_style == "antigravity" else constants.GEMINI_CLI_HEADERS
            headers = {
                **headers,
                "Authorization": f"Bearer {auth.access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }

            if is_thinking_model(self.model) and is_claude_model(self.model):
                headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

            async with httpx.AsyncClient(timeout=120.0) as client:
                for endpoint in constants.ANTIGRAVITY_ENDPOINT_FALLBACKS:
                    url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"

                    try:
                        async with _ANTIGRAVITY_HTTP_SEMAPHORE:
                            async with client.stream("POST", url, headers=headers, json=body) as response:
                                last_status_by_endpoint[endpoint] = response.status_code

                                # Retryable server-side issues
                                if response.status_code == 429 or response.status_code >= 500:
                                    base = min(8.0, 0.5 * (2**attempt))
                                    await asyncio.sleep(_retry_after_seconds(response, default=base))
                                    continue

                                # Some endpoints (especially sandbox) may return 403 while prod works.
                                # Keep trying other endpoints instead of failing fast.
                                if response.status_code in (401, 403, 404):
                                    # Capture a representative error for diagnostics
                                    try:
                                        response.raise_for_status()
                                    except httpx.HTTPStatusError as e:
                                        last_http_error = e
                                    continue

                                response.raise_for_status()

                                async for line in response.aiter_lines():
                                    if line.startswith("data: "):
                                        data = line[6:]
                                        if data == "[DONE]":
                                            return

                                        try:
                                            chunk_data = json.loads(data)
                                            inner = chunk_data.get("response", chunk_data)
                                            candidates = inner.get("candidates", [])

                                            if candidates:
                                                parts = candidates[0].get("content", {}).get("parts", [])
                                                for part in parts:
                                                    if "text" in part and not part.get("thought"):
                                                        yield ChatGenerationChunk(
                                                            message=AIMessageChunk(content=part["text"])
                                                        )
                                        except json.JSONDecodeError:
                                            continue

                                return

                    except httpx.HTTPStatusError as e:
                        last_http_error = e
                        # Only treat rate limits / transient server failures as retryable per-endpoint
                        if e.response.status_code in (429, 500, 502, 503, 504):
                            base = min(8.0, 0.5 * (2**attempt))
                            await asyncio.sleep(_retry_after_seconds(e.response, default=base))
                            continue
                        # For other 4xx errors, try next endpoint
                        continue
                    except httpx.RequestError:
                        continue

            # If we got auth-like failures, force refresh once then retry.
            if attempt == 0:
                did_retry = False
                if last_http_error is not None and last_http_error.response.status_code in (401, 403):
                    try:
                        if self.auth is not None:
                            self.auth = await antigravity_auth.refresh_access_token(self.auth)
                            did_retry = True
                    except Exception:
                        pass

                project_override = self.project_id or _env_project_override()
                auth_project = _resolve_auth_project_id(self.auth)
                needs_project_refresh = (
                    not did_retry
                    and not project_override
                    and self.auth is not None
                    and (not auth_project or auth_project == constants.ANTIGRAVITY_DEFAULT_PROJECT_ID)
                    and any(status in (403, 429) for status in last_status_by_endpoint.values())
                )
                if needs_project_refresh:
                    try:
                        project_id = await antigravity_auth.fetch_project_id(self.auth.access_token)
                        if project_id:
                            self.auth.project_id = project_id
                            did_retry = True
                    except Exception:
                        pass

                if did_retry:
                    continue

            # If we were primarily rate-limited, wait a bit and retry.
            if any(status == 429 for status in last_status_by_endpoint.values()):
                await asyncio.sleep(min(10.0, 1.0 * (2**attempt)))
                continue

        details = ", ".join(f"{k}={v}" for k, v in last_status_by_endpoint.items())
        if last_http_error is not None:
            # If any endpoint rate-limited us, surface 429 even if the last HTTP error was a sandbox 403.
            code = 429 if any(status == 429 for status in last_status_by_endpoint.values()) else last_http_error.response.status_code
            # Avoid huge payloads, but include a small hint.
            body_snippet = (await _read_response_text(last_http_error.response)).strip()
            if len(body_snippet) > 500:
                body_snippet = body_snippet[:500] + "…"
            raise RuntimeError(
                "Antigravity request failed for all endpoints. "
                f"Last status={code}. Endpoint statuses: {details}. "
                "If you see 401/403, re-authenticate via the Antigravity OAuth flow and ensure your account has access. "
                f"Response: {body_snippet}"
            ) from last_http_error

        raise RuntimeError(f"All Antigravity endpoints failed. Endpoint statuses: {details}")
