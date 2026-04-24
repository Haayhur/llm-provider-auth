"""
LangChain ChatModel for OpenAI Codex API.

This module provides a LangChain BaseChatModel that communicates with OpenAI's
ChatGPT Codex backend via OAuth authentication, allowing access to GPT-5.x
and Codex models using a ChatGPT subscription.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
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

from .codex_responses import (
    ResponsesSseAccumulator,
    latest_code_interpreter_container_id,
    latest_response_id,
    message_additional_input_items,
    normalize_followup_input_items,
    normalize_responses_message_content,
    normalize_responses_input_items,
    parse_responses_output,
    replayable_output_items,
    serialize_function_call_output,
    split_bound_tools,
)
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
_CODEX_OAUTH_ALLOWED_NON_CODEX_MODELS: frozenset[str] = frozenset({"gpt-5.2", "gpt-5.4", "gpt-5.4-mini"})


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
    - GPT-5.4 (low/medium/high/xhigh)
    - GPT-5.3 Codex (low/medium/high/xhigh)
    - GPT-5.2 models (none/low/medium/high/xhigh)
    - GPT-5.2 Codex (low/medium/high/xhigh)
    - GPT-5.1 Codex Max (low/medium/high/xhigh)
    - GPT-5.1 Codex (low/medium/high)
    - GPT-5.1 Codex Mini (medium/high)

    Example:
        ```python
        from langchain_antigravity import ChatCodex

        chat = ChatCodex(model="gpt-5.3-codex")
        response = chat.invoke("Hello!")
        ```
    """

    model: str = Field(default="gpt-5.3-codex")
    """Model name to use."""

    temperature: float | None = Field(default=None)
    """Sampling temperature."""

    max_tokens: int | None = Field(default=None)
    """Maximum output tokens."""

    reasoning_effort: str | None = Field(default=None)
    """Reasoning effort: none, low, medium, high, xhigh."""

    metadata: dict[str, Any] | None = Field(default=None)
    """Optional Responses API metadata payload."""

    include: list[str] | None = Field(default=None)
    """Optional Responses API include fields."""

    text: dict[str, Any] | None = Field(default=None)
    """Optional Responses API text configuration."""

    truncation: str | dict[str, Any] | None = Field(default=None)
    """Optional Responses API truncation mode/config."""

    parallel_tool_calls: bool | None = Field(default=None)
    """Optional Responses API parallel tool calling toggle."""

    stream_structured_events: bool = Field(default=False)
    """Emit non-text Responses events as empty-content stream chunks when enabled."""

    auto_execute_tools: bool = Field(default=True)
    """Execute bound function tools inside ChatCodex until the turn completes."""

    max_tool_rounds: int = Field(default=8)
    """Maximum number of internal function-tool execution rounds."""

    previous_response_id: str | None = Field(default=None)
    """Optional Responses API previous_response_id passthrough."""

    computer_executor: Callable[[dict[str, Any]], Any] | None = Field(default=None, exclude=True)
    """Optional computer-use host executor that returns a computer_call_output item."""

    computer_loop_max_steps: int = Field(default=8)
    """Maximum number of computer-use host execution steps."""

    computer_loop_timeout_seconds: float | None = Field(default=None)
    """Optional timeout per computer-use host execution step."""

    auth: CodexAuth | None = Field(default=None, exclude=True)
    """Authentication state. If not provided, will load from storage."""

    account_id: str | None = Field(default=None)
    """Optional ChatGPT account ID override."""

    session_id: str | None = Field(default=None, exclude=True)
    """Optional ChatGPT Codex session/thread id used via request headers."""

    extra_body: dict[str, Any] | None = Field(default=None, exclude=True)
    """Additional Responses API request fields to merge in."""

    on_auth_update: Callable[[CodexAuth], Any] | None = Field(default=None, exclude=True)
    """Optional callback invoked after auth refresh to persist updated tokens."""

    _tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _native_tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _bound_tools: dict[str, Any] = PrivateAttr(default_factory=dict)
    _default_input_items: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _reuse_previous_code_interpreter_container: bool = PrivateAttr(default=False)
    _tool_choice: str | dict[str, Any] | None = PrivateAttr(default=None)
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
        if tool_choice == "none":
            new_model = self.model_copy()
            new_model._tools = []
            new_model._native_tools = []
            new_model._bound_tools = {}
            new_model._tool_choice = None
            return new_model

        function_declarations, native_tools = split_bound_tools(tools)
        executable_tools: dict[str, Any] = {}
        for tool in tools:
            if isinstance(tool, dict) or isinstance(tool, type):
                continue
            if hasattr(tool, "name") and str(getattr(tool, "name", "") or "").strip():
                executable_tools[str(getattr(tool, "name")).strip()] = tool
                continue
            if callable(tool):
                name = str(getattr(tool, "__name__", "") or "").strip()
                if name:
                    executable_tools[name] = tool

        new_model = self.model_copy()
        new_model._tools = function_declarations
        new_model._native_tools = native_tools
        new_model._bound_tools = executable_tools
        if tool_choice in {"auto", "required"}:
            new_model._tool_choice = tool_choice
        else:
            new_model._tool_choice = None
        return new_model

    def bind_native_tools(
        self,
        native_tools: Sequence[dict[str, Any]],
        *,
        merge: bool = True,
    ) -> "ChatCodex":
        """Bind provider-native Responses API tools without coercing them to functions."""
        new_model = self.model_copy()
        if merge:
            new_model._native_tools = [*self._native_tools, *[dict(tool) for tool in native_tools if isinstance(tool, dict)]]
        else:
            new_model._native_tools = [dict(tool) for tool in native_tools if isinstance(tool, dict)]
        new_model._bound_tools = dict(self._bound_tools)
        return new_model

    def bind_file_search_tools(
        self,
        *,
        vector_store_ids: Sequence[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = None,
        include_results: bool = False,
        merge: bool = True,
    ) -> "ChatCodex":
        tool: dict[str, Any] = {"type": "file_search", "vector_store_ids": list(vector_store_ids)}
        if filters is not None:
            tool["filters"] = filters
        if max_num_results is not None:
            tool["max_num_results"] = max_num_results
        model = self.bind_native_tools([tool], merge=merge)
        if include_results:
            include = list(model.include or [])
            if "file_search_call.results" not in include:
                include.append("file_search_call.results")
            model.include = include
        return model

    def bind_code_interpreter_tools(
        self,
        *,
        container: str | dict[str, Any] | None = None,
        reuse_previous_container: bool = False,
        merge: bool = True,
    ) -> "ChatCodex":
        tool: dict[str, Any] = {"type": "code_interpreter"}
        if container is not None:
            tool["container"] = container
        model = self.bind_native_tools([tool], merge=merge)
        model._reuse_previous_code_interpreter_container = reuse_previous_container
        return model

    def bind_computer_use_tools(
        self,
        *,
        display_width: int | None = None,
        display_height: int | None = None,
        environment: str | None = None,
        merge: bool = True,
        **config: Any,
    ) -> "ChatCodex":
        tool: dict[str, Any] = {"type": "computer_use_preview"}
        if display_width is not None and display_height is not None:
            tool["display_width"] = display_width
            tool["display_height"] = display_height
        if environment is not None:
            tool["environment"] = environment
        tool.update({key: value for key, value in config.items() if value is not None})
        model = self.bind_native_tools([tool], merge=merge)
        if model.truncation is None:
            model.truncation = "auto"
        return model

    def bind_mcp_tools(
        self,
        tools: Sequence[dict[str, Any]] | dict[str, Any],
        *,
        merge: bool = True,
    ) -> "ChatCodex":
        if isinstance(tools, dict):
            tool_list = [tools]
        else:
            tool_list = [dict(tool) for tool in tools if isinstance(tool, dict)]
        normalized_tools = []
        for tool in tool_list:
            normalized = dict(tool)
            normalized.setdefault("type", "mcp")
            normalized_tools.append(normalized)
        return self.bind_native_tools(normalized_tools, merge=merge)

    def with_input_items(
        self,
        items: Sequence[dict[str, Any]],
        *,
        merge: bool = True,
    ) -> "ChatCodex":
        new_model = self.model_copy()
        normalized_items = [dict(item) for item in items if isinstance(item, dict)]
        if merge:
            new_model._default_input_items = [*self._default_input_items, *normalized_items]
        else:
            new_model._default_input_items = normalized_items
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

    def _subscription_error_message(
        self,
        *,
        error_code: str | None,
        error_message: str,
    ) -> str | None:
        """Return a user-friendly plan/subscription error when applicable."""
        combined = f"{error_code or ''} {error_message or ''}".lower()

        if "usage_not_included" in combined:
            return (
                "To use Codex with your ChatGPT plan, upgrade to Plus: "
                "https://chatgpt.com/explore/plus."
            )

        subscription_tokens = (
            "requires pro",
            "chatgpt pro",
            "pro subscription",
            "plan does not include",
            "not included in your plan",
            "subscription required",
            "upgrade to plus",
            "upgrade to pro",
        )
        if any(token in combined for token in subscription_tokens):
            return (
                "To use Codex with your ChatGPT plan, upgrade to Plus: "
                "https://chatgpt.com/explore/plus."
            )

        return None

    def _known_error_code_message(self, *, error_code: str | None, message: str) -> str | None:
        code = (error_code or "").strip().lower()
        if not code:
            return None
        if code == "context_length_exceeded":
            return "Input exceeds context window of this model."
        if code == "insufficient_quota":
            return "Quota exceeded. Check your plan and billing details."
        if code == "invalid_prompt":
            cleaned = (message or "").strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                return "Invalid prompt."
            return cleaned or "Invalid prompt."
        return None

    def _normalize_error_text(
        self,
        *,
        model: str,
        status_code: int | None,
        error_text: str,
    ) -> str:
        """Normalize provider errors into clearer, user-facing messages."""
        message = error_text
        error_code: str | None = None

        try:
            payload = json.loads(error_text)
            if isinstance(payload, dict):
                error_obj = payload.get("error")
                if isinstance(error_obj, dict):
                    code = error_obj.get("code")
                    if isinstance(code, str) and code.strip():
                        error_code = code
                    raw_msg = error_obj.get("message")
                    if isinstance(raw_msg, str) and raw_msg.strip():
                        message = raw_msg
                elif isinstance(error_obj, str) and error_obj.strip():
                    message = error_obj

                if not error_code:
                    top_code = payload.get("code")
                    if isinstance(top_code, str) and top_code.strip():
                        error_code = top_code
        except Exception:
            pass

        known_code_message = self._known_error_code_message(error_code=error_code, message=message)
        if known_code_message:
            return known_code_message

        subscription_message = self._subscription_error_message(
            error_code=error_code,
            error_message=message,
        )
        if subscription_message:
            return subscription_message
        return message

    def _is_allowed_codex_oauth_model(self, model: str) -> bool:
        model_id = (model or "").strip().lower()
        if not model_id:
            return False
        if "codex" in model_id:
            return True
        return model_id in _CODEX_OAUTH_ALLOWED_NON_CODEX_MODELS

    def _should_apply_default_reasoning(self, model: str) -> bool:
        model_id = (model or "").strip().lower()
        if "gpt-5" not in model_id:
            return False
        if "gpt-5-chat" in model_id:
            return False
        if "gpt-5-pro" in model_id:
            return False
        return True

    def _extract_error_message(self, response: httpx.Response, *, model: str) -> str:
        return self._normalize_error_text(
            model=model,
            status_code=response.status_code,
            error_text=response.text,
        )

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
        accumulator = ResponsesSseAccumulator()
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
            if isinstance(evt, dict):
                accumulator.add_event(evt)
        return accumulator.build_response()

    def _raise_for_responses_payload_error(
        self,
        response_data: dict[str, Any],
        *,
        model: str,
    ) -> None:
        status = str(response_data.get("status") or "").strip().lower()
        error = response_data.get("error")
        if status == "failed" or error is not None:
            error_text = error if isinstance(error, str) else json.dumps({"error": error}, default=str)
            message = self._normalize_error_text(
                model=model,
                status_code=None,
                error_text=error_text,
            )
            raise RuntimeError(f"Codex API error: {message}")
        incomplete_details = response_data.get("incomplete_details")
        if status == "incomplete" or incomplete_details is not None:
            reason = ""
            if isinstance(incomplete_details, dict):
                reason = str(incomplete_details.get("reason") or "").strip()
            if not reason:
                reason = "unknown"
            raise RuntimeError(f"Codex API incomplete response: {reason}")

    def _parse_responses_api_output(self, response_data: dict[str, Any]) -> AIMessage:
        """Parse Responses-style payload into an AIMessage."""
        parsed = parse_responses_output(response_data)
        response_metadata: dict[str, Any] = {}
        if parsed.reasoning:
            response_metadata["reasoning"] = parsed.reasoning
        if parsed.codex_items:
            response_metadata["codex_items"] = parsed.codex_items
        if parsed.native_tool_events:
            response_metadata["native_tool_events"] = parsed.native_tool_events
        if parsed.file_search_results:
            response_metadata["file_search_results"] = parsed.file_search_results
        if parsed.code_interpreter:
            response_metadata["code_interpreter"] = parsed.code_interpreter
        if parsed.computer_calls:
            response_metadata["computer_calls"] = parsed.computer_calls
        if parsed.mcp_calls:
            response_metadata["mcp_calls"] = parsed.mcp_calls
        if parsed.mcp_approvals:
            response_metadata["mcp_approvals"] = parsed.mcp_approvals
        if parsed.response_id:
            response_metadata["response_id"] = parsed.response_id
        if parsed.raw_output_items:
            response_metadata["codex_output_items"] = parsed.raw_output_items

        return AIMessage(
            content=parsed.content,
            tool_calls=[
                ToolCall(name=tool_call["name"], args=tool_call["args"], id=tool_call["id"])
                for tool_call in parsed.external_tool_calls
            ],
            additional_kwargs={"codex_output_items": parsed.raw_output_items} if parsed.raw_output_items else None,
            response_metadata=response_metadata or None,
        )

    def _merge_combined_output_items(
        self,
        message: AIMessage,
        output_items: list[dict[str, Any]],
        handled_tool_calls: list[dict[str, Any]] | None = None,
        handled_computer_calls: list[dict[str, Any]] | None = None,
    ) -> AIMessage:
        parsed = parse_responses_output({"output": output_items})
        response_metadata = dict(getattr(message, "response_metadata", {}) or {})
        if parsed.reasoning:
            response_metadata["reasoning"] = parsed.reasoning
        if parsed.codex_items:
            response_metadata["codex_items"] = parsed.codex_items
        if parsed.native_tool_events:
            response_metadata["native_tool_events"] = parsed.native_tool_events
        if parsed.file_search_results:
            response_metadata["file_search_results"] = parsed.file_search_results
        if parsed.code_interpreter:
            response_metadata["code_interpreter"] = parsed.code_interpreter
        if parsed.computer_calls:
            response_metadata["computer_calls"] = parsed.computer_calls
        if parsed.mcp_calls:
            response_metadata["mcp_calls"] = parsed.mcp_calls
        if parsed.mcp_approvals:
            response_metadata["mcp_approvals"] = parsed.mcp_approvals
        if parsed.response_id:
            response_metadata["response_id"] = parsed.response_id
        if handled_tool_calls:
            response_metadata["handled_tool_calls"] = handled_tool_calls
        if handled_computer_calls:
            response_metadata["handled_computer_calls"] = handled_computer_calls
        if output_items:
            response_metadata["codex_output_items"] = output_items
        additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
        if output_items:
            additional_kwargs["codex_output_items"] = output_items
        update = {
            "tool_calls": [],
            "additional_kwargs": additional_kwargs or None,
            "response_metadata": response_metadata or None,
        }
        if hasattr(message, "model_copy"):
            return message.model_copy(update=update)
        return AIMessage(
            content=message.content,
            tool_calls=[],
            usage_metadata=getattr(message, "usage_metadata", None) or None,
            additional_kwargs=additional_kwargs or None,
            response_metadata=response_metadata or None,
        )

    def _tool_call_signature(self, tool_call: dict[str, Any]) -> str:
        name = str(tool_call.get("name") or "").strip()
        args = tool_call.get("args") or {}
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"

    def _resolve_previous_response_id(
        self,
        messages: Sequence[BaseMessage],
        explicit_previous_response_id: str | None = None,
    ) -> str | None:
        if explicit_previous_response_id is not None:
            value = str(explicit_previous_response_id or "").strip()
            return value or None
        derived = latest_response_id(messages)
        if derived:
            return derived
        configured = str(self.previous_response_id or "").strip()
        return configured or None

    def _resolve_session_id(self, explicit_session_id: str | None = None) -> str:
        if explicit_session_id is not None:
            value = str(explicit_session_id or "").strip()
            if value:
                self.session_id = value
                return value
        configured = str(self.session_id or "").strip()
        if configured:
            return configured
        generated = str(uuid.uuid4())
        self.session_id = generated
        return generated

    def _apply_session_headers(self, headers: dict[str, str], session_id: str) -> None:
        if not session_id:
            return
        headers["session_id"] = session_id
        headers["x-client-request-id"] = session_id

    def _resolve_request_tools(self, messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
        tools = [dict(tool) for tool in [*self._tools, *self._native_tools]]
        if not self._reuse_previous_code_interpreter_container:
            return tools
        container_id = latest_code_interpreter_container_id(messages)
        if not container_id:
            return tools
        resolved_tools: list[dict[str, Any]] = []
        for tool in tools:
            if str(tool.get("type") or "").strip() == "code_interpreter":
                resolved_tool = dict(tool)
                resolved_tool["container"] = container_id
                resolved_tools.append(resolved_tool)
                continue
            resolved_tools.append(tool)
        return resolved_tools

    def _apply_request_context_metadata(
        self,
        message: AIMessage,
        *,
        previous_response_id_used: str | None = None,
    ) -> AIMessage:
        if not previous_response_id_used:
            return message
        response_metadata = dict(getattr(message, "response_metadata", {}) or {})
        response_metadata["previous_response_id_used"] = previous_response_id_used
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"response_metadata": response_metadata or None})
        return AIMessage(
            content=message.content,
            tool_calls=getattr(message, "tool_calls", None) or [],
            usage_metadata=getattr(message, "usage_metadata", None) or None,
            additional_kwargs=getattr(message, "additional_kwargs", None) or None,
            response_metadata=response_metadata or None,
        )

    def _computer_call_signature(self, computer_call: dict[str, Any]) -> str:
        action = computer_call.get("action")
        call_id = str(computer_call.get("call_id") or computer_call.get("id") or "").strip()
        return f"{call_id}:{json.dumps(action, sort_keys=True, default=str)}"

    def _has_assistant_message_item(self, output_items: list[dict[str, Any]]) -> bool:
        return any(
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role", "assistant") == "assistant"
            for item in output_items
        )

    async def _invoke_bound_tool(self, tool: Any, args: dict[str, Any]) -> Any:
        if hasattr(tool, "ainvoke") and callable(getattr(tool, "ainvoke")):
            return await tool.ainvoke(args)
        if hasattr(tool, "invoke") and callable(getattr(tool, "invoke")):
            return tool.invoke(args)
        if hasattr(tool, "arun") and callable(getattr(tool, "arun")):
            return await tool.arun(**args)
        if hasattr(tool, "run") and callable(getattr(tool, "run")):
            return tool.run(**args)
        if callable(tool):
            try:
                result = tool(**args)
            except TypeError:
                result = tool(args)
            if inspect.isawaitable(result):
                return await result
            return result
        raise RuntimeError(f"Bound tool is not executable: {tool!r}")

    async def _invoke_computer_executor(self, computer_call: dict[str, Any]) -> dict[str, Any]:
        if self.computer_executor is None:
            raise RuntimeError("Codex computer_use_preview requested without a configured computer_executor")
        result = self.computer_executor(computer_call)
        if inspect.isawaitable(result):
            if self.computer_loop_timeout_seconds is not None:
                result = await asyncio.wait_for(result, timeout=self.computer_loop_timeout_seconds)
            else:
                result = await result
        if self.computer_loop_timeout_seconds is not None and not inspect.isawaitable(result):
            pass
        if not isinstance(result, dict):
            raise RuntimeError("computer_executor must return a dict representing computer_call_output")

        call_id = str(computer_call.get("call_id") or computer_call.get("id") or "").strip()
        if result.get("type") == "computer_call_output":
            output_item = dict(result)
            output_item.setdefault("call_id", call_id)
            return output_item

        output_payload = dict(result)
        if "output" not in output_payload:
            if "image_url" in output_payload or "file_id" in output_payload:
                output_payload = {"output": output_payload}
            else:
                raise RuntimeError("computer_executor result must contain 'output' or screenshot fields")
        return {
            "type": "computer_call_output",
            "call_id": call_id,
            **output_payload,
        }

    async def _execute_bound_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        output_items: list[dict[str, Any]] = []
        handled_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name") or "").strip()
            call_id = str(tool_call.get("id") or "").strip()
            if not tool_name or not call_id:
                continue
            tool = self._bound_tools.get(tool_name)
            if tool is None:
                raise RuntimeError(f"Codex auto_execute_tools missing bound implementation for '{tool_name}'")
            success = True
            try:
                result = await self._invoke_bound_tool(tool, dict(tool_call.get("args") or {}))
            except Exception as exc:
                success = False
                result = {"error": str(exc)}
            output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": serialize_function_call_output(result),
                }
            )
            handled_tool_calls.append(
                {
                    "id": call_id,
                    "name": tool_name,
                    "args": dict(tool_call.get("args") or {}),
                    "output": result,
                    "success": success,
                }
            )
        return output_items, handled_tool_calls

    async def _execute_computer_calls(
        self,
        computer_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        output_items: list[dict[str, Any]] = []
        handled_calls: list[dict[str, Any]] = []
        for computer_call in computer_calls:
            call_id = str(computer_call.get("call_id") or computer_call.get("id") or "").strip()
            success = True
            try:
                output_item = await self._invoke_computer_executor(computer_call)
            except Exception as exc:
                success = False
                output_item = {
                    "type": "computer_call_output",
                    "call_id": call_id,
                    "output": {"error": str(exc)},
                }
            output_items.append(output_item)
            handled_calls.append(
                {
                    "id": call_id,
                    "action": computer_call.get("action"),
                    "output": output_item.get("output"),
                    "success": success,
                }
            )
        return output_items, handled_calls

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert LangChain messages to ChatGPT Codex backend `input` format."""
        input_items: list[dict[str, Any]] = []

        def _tool_call_parts(tool_call: Any) -> tuple[str, str, str]:
            name = ""
            call_id = ""
            args: Any = {}
            if isinstance(tool_call, dict):
                function = tool_call.get("function")
                if isinstance(function, dict):
                    name = str(function.get("name") or "").strip()
                    args = function.get("arguments")
                name = name or str(tool_call.get("name") or "").strip()
                args = tool_call.get("args", args)
                call_id = str(tool_call.get("call_id") or tool_call.get("tool_call_id") or tool_call.get("id") or "").strip()
            else:
                name = str(getattr(tool_call, "name", "") or "").strip()
                args = getattr(tool_call, "args", {})
                call_id = str(
                    getattr(tool_call, "call_id", None)
                    or getattr(tool_call, "tool_call_id", None)
                    or getattr(tool_call, "id", "")
                    or ""
                ).strip()
            if isinstance(args, str):
                args_str = args
            else:
                args_str = json.dumps(args or {}, default=str)
            return name, call_id, args_str

        for msg in messages:
            if isinstance(msg, SystemMessage):
                input_items.append({
                    "type": "message",
                    "role": "developer",
                    "content": normalize_responses_message_content(msg.content),
                })
                input_items.extend(message_additional_input_items(msg))
            elif isinstance(msg, HumanMessage):
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": normalize_responses_message_content(msg.content),
                })
                input_items.extend(message_additional_input_items(msg))
            elif isinstance(msg, AIMessage):
                replay_items = replayable_output_items(msg)
                if replay_items:
                    input_items.extend(replay_items)
                    has_message_item = any(
                        isinstance(item, dict)
                        and item.get("type") == "message"
                        and item.get("role") == "assistant"
                        for item in replay_items
                    )
                    has_tool_call_item = any(
                        isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}
                        for item in replay_items
                    )
                else:
                    has_message_item = False
                    has_tool_call_item = False

                content = msg.content or ""
                if content and not has_message_item:
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": normalize_responses_message_content(content),
                    })
                if not has_tool_call_item:
                    for tool_call in list(getattr(msg, "tool_calls", None) or []):
                        name, call_id, args_str = _tool_call_parts(tool_call)
                        if not name or not call_id:
                            continue
                        input_items.append({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": args_str,
                        })
                input_items.extend(message_additional_input_items(msg))
            elif isinstance(msg, ToolMessage):
                tool_call_id = str(getattr(msg, "tool_call_id", "") or "").strip()
                if tool_call_id:
                    additional_kwargs = getattr(msg, "additional_kwargs", None)
                    if isinstance(additional_kwargs, dict) and "codex_function_call_output" in additional_kwargs:
                        output = additional_kwargs["codex_function_call_output"]
                    else:
                        output = serialize_function_call_output(msg.content)
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": output,
                    })
                elif msg.content:
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": normalize_responses_message_content(msg.content),
                    })
                input_items.extend(message_additional_input_items(msg))

        normalized_items = normalize_responses_input_items(input_items)
        # `with_input_items(...)` is the exact-input escape hatch used by the
        # Responses proxy. Preserve those items verbatim so follow-up payloads
        # such as standalone `function_call_output` entries are not dropped.
        normalized_items.extend(normalize_followup_input_items(self._default_input_items))
        return normalized_items

    def _build_request_body(
        self,
        input_items: list[dict[str, Any]],
        instructions: str,
        stream: bool = False,
        *,
        metadata: dict[str, Any] | None = None,
        include: list[str] | None = None,
        text: dict[str, Any] | None = None,
        truncation: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build Codex API request body."""
        effective_model = normalize_codex_model(self.model)
        if not self._is_allowed_codex_oauth_model(effective_model):
            raise ValueError(
                f"Model '{effective_model}' is not available in Codex OAuth flow. "
                "Use a Codex model or one of: gpt-5.2, gpt-5.4."
            )

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
            body["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
        elif self._should_apply_default_reasoning(effective_model):
            body["reasoning"] = {"effort": "medium", "summary": "auto"}

        # Codex API does not support the metadata parameter
        # so we ignore metadata_payload to prevent 400 errors.
        include_payload = include if include is not None else self.include
        if include_payload is not None:
            body["include"] = include_payload
        text_payload = text if text is not None else self.text
        if text_payload is not None:
            body["text"] = text_payload
        truncation_payload = truncation if truncation is not None else self.truncation
        if truncation_payload is not None:
            body["truncation"] = truncation_payload
        parallel_tool_calls_payload = (
            parallel_tool_calls if parallel_tool_calls is not None else self.parallel_tool_calls
        )
        if parallel_tool_calls_payload is not None:
            body["parallel_tool_calls"] = parallel_tool_calls_payload
        # ChatGPT Codex thread continuity is anchored by request headers (`session_id`),
        # not a `previous_response_id` field in the JSON body. We still accept the
        # parameter at the model boundary for compatibility and metadata bookkeeping.
        _ = previous_response_id

        request_tools = [dict(tool) for tool in tools] if tools is not None else [*self._tools, *self._native_tools]
        if request_tools:
            body["tools"] = request_tools
            if self._tool_choice is not None:
                body["tool_choice"] = self._tool_choice

        merged_extra_body = dict(self.extra_body or {})
        if extra_body:
            merged_extra_body.update(extra_body)
        if merged_extra_body:
            reserved_keys = {
                "model",
                "instructions",
                "store",
                "stream",
                "input",
                "tools",
                "tool_choice",
            }
            for key, value in merged_extra_body.items():
                if key in reserved_keys:
                    continue
                body[key] = value

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
                        "additional_kwargs": getattr(base_message, "additional_kwargs", None) or None,
                        "response_metadata": response_metadata or None,
                    }
                )
            return AIMessage(
                content=base_message.content,
                tool_calls=base_message.tool_calls,
                usage_metadata=usage_metadata or None,
                additional_kwargs=getattr(base_message, "additional_kwargs", None) or None,
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
        resolved_previous_response_id = self._resolve_previous_response_id(
            messages,
            explicit_previous_response_id=kwargs.get("previous_response_id"),
        )
        request_session_id = self._resolve_session_id(kwargs.get("session_id"))
        resolved_tools = self._resolve_request_tools(messages)
        effective_model = normalize_codex_model(self.model)
        instructions = get_codex_instructions(effective_model)
        body_kwargs = {
            "metadata": kwargs.get("metadata"),
            "include": kwargs.get("include"),
            "text": kwargs.get("text"),
            "truncation": kwargs.get("truncation"),
            "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
            "previous_response_id": resolved_previous_response_id,
            "tools": resolved_tools,
            "extra_body": kwargs.get("extra_body") or kwargs.get("responses_api_extra"),
        }

        headers = {
            **CODEX_HEADERS,
            "Authorization": f"Bearer {auth.access_token}",
            "Content-Type": "application/json",
            # Codex backend returns SSE even for non-stream; match Codex CLI behavior.
            "Accept": "text/event-stream",
        }

        if auth.account_id:
            headers["chatgpt-account-id"] = auth.account_id
        self._apply_session_headers(headers, request_session_id)

        # IMPORTANT: Codex uses /codex/responses (not /responses).
        url = f"{CODEX_BASE_URL}/codex/responses"

        async with httpx.AsyncClient(timeout=120.0) as client:
            combined_output_items: list[dict[str, Any]] = []
            handled_tool_calls: list[dict[str, Any]] = []
            handled_computer_calls: list[dict[str, Any]] = []
            recent_tool_signatures: set[str] = set()
            recent_computer_signatures: set[str] = set()
            computer_steps = 0
            for tool_round in range(self.max_tool_rounds + 1):
                body = self._build_request_body(
                    input_items,
                    instructions,
                    stream=False,
                    **body_kwargs,
                )
                for attempt in range(2):
                    response = await client.post(url, headers=headers, json=body)

                    if not response.is_success:
                        error_text = self._extract_error_message(response, model=effective_model)
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
                        response_data = final_response
                    else:
                        try:
                            response_data = response.json()
                        except json.JSONDecodeError:
                            final_response = self._parse_sse_final_response(response.text)
                            if not final_response:
                                raise
                            response_data = final_response

                    self._raise_for_responses_payload_error(response_data, model=effective_model)
                    message = self._parse_response(response_data)
                    if "output" in response_data and (self.auto_execute_tools or self.computer_executor is not None):
                        parsed = parse_responses_output(response_data)
                        if self._has_assistant_message_item(parsed.raw_output_items):
                            recent_tool_signatures.clear()
                            recent_computer_signatures.clear()
                        if parsed.raw_output_items:
                            combined_output_items.extend(parsed.raw_output_items)
                        follow_up_items: list[dict[str, Any]] = []
                        if parsed.computer_calls and self.computer_executor is not None:
                            computer_steps += len(parsed.computer_calls)
                            if computer_steps > self.computer_loop_max_steps:
                                raise RuntimeError(
                                    f"Codex computer_use_preview exceeded computer_loop_max_steps={self.computer_loop_max_steps}"
                                )
                            for computer_call in parsed.computer_calls:
                                signature = self._computer_call_signature(computer_call)
                                if signature in recent_computer_signatures:
                                    raise RuntimeError(
                                        "Codex computer_use_preview detected repeated computer call without new output"
                                    )
                                recent_computer_signatures.add(signature)
                            computer_output_items, handled_computer_batch = await self._execute_computer_calls(parsed.computer_calls)
                            combined_output_items.extend(computer_output_items)
                            handled_computer_calls.extend(handled_computer_batch)
                            follow_up_items.extend(computer_output_items)
                        if self.auto_execute_tools and parsed.external_tool_calls:
                            for tool_call in parsed.external_tool_calls:
                                signature = self._tool_call_signature(tool_call)
                                if signature in recent_tool_signatures:
                                    raise RuntimeError(
                                        f"Codex auto_execute_tools detected repeated tool call without assistant progress: {tool_call['name']}"
                                    )
                                recent_tool_signatures.add(signature)
                            tool_output_items, handled_batch = await self._execute_bound_tool_calls(parsed.external_tool_calls)
                            combined_output_items.extend(tool_output_items)
                            handled_tool_calls.extend(handled_batch)
                            follow_up_items.extend(tool_output_items)
                        if follow_up_items:
                            input_items.extend(parsed.raw_output_items)
                            input_items.extend(follow_up_items)
                            break
                    llm_output = {"model_name": self.model}
                    if message.usage_metadata:
                        llm_output["token_usage"] = _normalize_token_usage(message.usage_metadata)
                    if combined_output_items:
                        message = self._merge_combined_output_items(
                            message,
                            combined_output_items,
                            handled_tool_calls=handled_tool_calls,
                            handled_computer_calls=handled_computer_calls,
                        )
                    message = self._apply_request_context_metadata(
                        message,
                        previous_response_id_used=resolved_previous_response_id,
                    )
                    return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)
                else:
                    raise RuntimeError("Codex API error: unauthorized after token refresh")
                continue
            raise RuntimeError(f"Codex auto_execute_tools exceeded max_tool_rounds={self.max_tool_rounds}")

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
        resolved_previous_response_id = self._resolve_previous_response_id(
            messages,
            explicit_previous_response_id=kwargs.get("previous_response_id"),
        )
        request_session_id = self._resolve_session_id(kwargs.get("session_id"))
        resolved_tools = self._resolve_request_tools(messages)
        effective_model = normalize_codex_model(self.model)
        instructions = get_codex_instructions(effective_model)
        emit_structured_events = bool(kwargs.get("stream_structured_events", self.stream_structured_events))
        body_kwargs = {
            "metadata": kwargs.get("metadata"),
            "include": kwargs.get("include"),
            "text": kwargs.get("text"),
            "truncation": kwargs.get("truncation"),
            "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
            "previous_response_id": resolved_previous_response_id,
            "tools": resolved_tools,
            "extra_body": kwargs.get("extra_body") or kwargs.get("responses_api_extra"),
        }

        headers = {
            **CODEX_HEADERS,
            "Authorization": f"Bearer {auth.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        if auth.account_id:
            headers["chatgpt-account-id"] = auth.account_id
        self._apply_session_headers(headers, request_session_id)

        url = f"{CODEX_BASE_URL}/codex/responses"

        async with httpx.AsyncClient(timeout=120.0) as client:
            unauthorized_retry_attempted = False
            handled_tool_calls: list[dict[str, Any]] = []
            handled_computer_calls: list[dict[str, Any]] = []
            recent_tool_signatures: set[str] = set()
            recent_computer_signatures: set[str] = set()
            computer_steps = 0
            for tool_round in range(self.max_tool_rounds + 1):
                body = self._build_request_body(
                    input_items,
                    instructions,
                    stream=True,
                    **body_kwargs,
                )
                continue_tool_loop = False
                for attempt in range(2):
                    request_saw_delta = False
                    async with client.stream("POST", url, headers=headers, json=body) as response:
                        accumulator = ResponsesSseAccumulator()
                        saw_done = False
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
                            error_text = self._normalize_error_text(
                                model=effective_model,
                                status_code=response.status_code,
                                error_text=error_text,
                            )
                            raise RuntimeError(f"Codex API error: {response.status_code} - {error_text}")

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    saw_done = True
                                    break

                                try:
                                    chunk_data = json.loads(data)
                                    if isinstance(chunk_data, dict):
                                        accumulator.add_event(chunk_data)
                                    evt_type = chunk_data.get("type")

                                    if evt_type in ("response.output_text.delta", "response.output_text.fragment"):
                                        delta_text = chunk_data.get("delta") or chunk_data.get("text")
                                        if isinstance(delta_text, str) and delta_text:
                                            request_saw_delta = True
                                            yield ChatGenerationChunk(message=AIMessageChunk(content=delta_text))
                                            continue

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
                                                            request_saw_delta = True
                                                            yield ChatGenerationChunk(
                                                                message=AIMessageChunk(content=text)
                                                            )
                                            continue

                                    if evt_type in (
                                        "response.reasoning_summary_text.delta",
                                        "response.reasoning_text.delta",
                                    ):
                                        delta_text = chunk_data.get("delta") or chunk_data.get("text")
                                        if isinstance(delta_text, str) and delta_text:
                                            if emit_structured_events:
                                                yield ChatGenerationChunk(
                                                    message=AIMessageChunk(
                                                        content="",
                                                        additional_kwargs={
                                                            "codex_event": {"type": evt_type, "delta": delta_text}
                                                        },
                                                    )
                                                )
                                            callback = getattr(run_manager, "on_custom_event", None)
                                            if callable(callback):
                                                maybe_result = callback(
                                                    "codex.responses.reasoning.delta",
                                                    {"type": evt_type, "delta": delta_text},
                                                )
                                                if asyncio.iscoroutine(maybe_result):
                                                    await maybe_result
                                            continue

                                    if evt_type == "response.output_item.done":
                                        item = chunk_data.get("item")
                                        if isinstance(item, dict):
                                            if emit_structured_events:
                                                yield ChatGenerationChunk(
                                                    message=AIMessageChunk(
                                                        content="",
                                                        additional_kwargs={"codex_event": {"type": evt_type, "item": item}},
                                                    )
                                                )
                                            callback = getattr(run_manager, "on_custom_event", None)
                                            if callable(callback):
                                                maybe_result = callback(
                                                    "codex.responses.output_item.done",
                                                    {"item": item},
                                                )
                                                if asyncio.iscoroutine(maybe_result):
                                                    await maybe_result
                                except json.JSONDecodeError:
                                    continue
                        final_response = accumulator.build_response()
                        if final_response:
                            if emit_structured_events:
                                yield ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content="",
                                        additional_kwargs={
                                            "codex_event": {
                                                "type": "response.completed",
                                                "response": final_response,
                                            }
                                        },
                                    )
                                )
                            callback = getattr(run_manager, "on_custom_event", None)
                            if callable(callback):
                                maybe_result = callback(
                                    "codex.responses.completed",
                                    {"response": final_response},
                                )
                                if asyncio.iscoroutine(maybe_result):
                                    await maybe_result
                            self._raise_for_responses_payload_error(final_response, model=effective_model)
                            if "output" in final_response and (self.auto_execute_tools or self.computer_executor is not None):
                                parsed = parse_responses_output(final_response)
                                if self._has_assistant_message_item(parsed.raw_output_items):
                                    recent_tool_signatures.clear()
                                    recent_computer_signatures.clear()
                                follow_up_items: list[dict[str, Any]] = []
                                if parsed.computer_calls and self.computer_executor is not None:
                                    computer_steps += len(parsed.computer_calls)
                                    if computer_steps > self.computer_loop_max_steps:
                                        raise RuntimeError(
                                            f"Codex computer_use_preview exceeded computer_loop_max_steps={self.computer_loop_max_steps}"
                                        )
                                    for computer_call in parsed.computer_calls:
                                        signature = self._computer_call_signature(computer_call)
                                        if signature in recent_computer_signatures:
                                            raise RuntimeError(
                                                "Codex computer_use_preview detected repeated computer call without new output"
                                            )
                                        recent_computer_signatures.add(signature)
                                    computer_output_items, handled_computer_batch = await self._execute_computer_calls(parsed.computer_calls)
                                    handled_computer_calls.extend(handled_computer_batch)
                                    if emit_structured_events:
                                        for item in computer_output_items:
                                            yield ChatGenerationChunk(
                                                message=AIMessageChunk(
                                                    content="",
                                                    additional_kwargs={
                                                        "codex_event": {
                                                            "type": "response.computer_call_output",
                                                            "item": item,
                                                        }
                                                    },
                                                )
                                            )
                                    callback = getattr(run_manager, "on_custom_event", None)
                                    if callable(callback):
                                        maybe_result = callback(
                                            "codex.responses.computer_call_output",
                                            {"items": computer_output_items, "handled_computer_calls": handled_computer_batch},
                                        )
                                        if asyncio.iscoroutine(maybe_result):
                                            await maybe_result
                                    follow_up_items.extend(computer_output_items)
                                if self.auto_execute_tools and parsed.external_tool_calls:
                                    for tool_call in parsed.external_tool_calls:
                                        signature = self._tool_call_signature(tool_call)
                                        if signature in recent_tool_signatures:
                                            raise RuntimeError(
                                                f"Codex auto_execute_tools detected repeated tool call without assistant progress: {tool_call['name']}"
                                            )
                                        recent_tool_signatures.add(signature)
                                    tool_output_items, handled_batch = await self._execute_bound_tool_calls(parsed.external_tool_calls)
                                    handled_tool_calls.extend(handled_batch)
                                    if emit_structured_events:
                                        for item in tool_output_items:
                                            yield ChatGenerationChunk(
                                                message=AIMessageChunk(
                                                    content="",
                                                    additional_kwargs={
                                                        "codex_event": {
                                                            "type": "response.function_call_output",
                                                            "item": item,
                                                        }
                                                    },
                                                )
                                            )
                                    callback = getattr(run_manager, "on_custom_event", None)
                                    if callable(callback):
                                        maybe_result = callback(
                                            "codex.responses.function_call_output",
                                            {"items": tool_output_items, "handled_tool_calls": handled_batch},
                                        )
                                        if asyncio.iscoroutine(maybe_result):
                                            await maybe_result
                                    follow_up_items.extend(tool_output_items)
                                if follow_up_items:
                                    input_items.extend(parsed.raw_output_items)
                                    input_items.extend(follow_up_items)
                                    continue_tool_loop = True
                                    break
                            return
                        if saw_done:
                            raise RuntimeError("Codex API stream ended with [DONE] but no final response payload")
                        if request_saw_delta:
                            _LOGGER.warning("Codex stream ended without [DONE]; returning partial response")
                            return
                        raise RuntimeError("Codex API stream ended unexpectedly before completion event")
                else:
                    raise RuntimeError("Codex API error: unauthorized after token refresh")
                if continue_tool_loop:
                    continue
                return
            raise RuntimeError(f"Codex auto_execute_tools exceeded max_tool_rounds={self.max_tool_rounds}")
