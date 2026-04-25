"""
LangChain ChatModel for GitHub Copilot via the Copilot CLI SDK.

This wraps the github-copilot-sdk client and exposes it as a LangChain BaseChatModel.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
from typing import Any, AsyncIterator, Callable, Sequence

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr

try:
    from .copilot_auth import CopilotAuth, load_copilot_auth_from_storage
    from .copilot_models import _copilot_api_base
except Exception:  # pragma: no cover
    from copilot_auth import CopilotAuth, load_copilot_auth_from_storage  # type: ignore
    from copilot_models import _copilot_api_base  # type: ignore


try:
    from copilot import CopilotClient, Tool, ToolResult, SessionEvent  # type: ignore
    from copilot.generated.session_events import SessionEventType  # type: ignore
except Exception:  # pragma: no cover
    CopilotClient = None  # type: ignore[assignment]
    SessionEvent = Any  # type: ignore[misc, assignment]
    SessionEventType = None  # type: ignore[assignment]
    ToolResult = dict[str, Any]  # type: ignore[misc, assignment]

    class Tool:  # type: ignore[no-redef]
        def __init__(
            self,
            *,
            name: str,
            description: str = "",
            handler: Callable[..., Any] | None = None,
            parameters: dict[str, Any] | None = None,
        ) -> None:
            self.name = name
            self.description = description
            self.handler = handler
            self.parameters = parameters or {}


def _patch_copilot_context_parsing() -> None:
    try:
        from copilot.generated import session_events as _session_events  # type: ignore
    except Exception:
        return

    if getattr(_session_events, "_context_patch_applied", False):
        return

    original_from_dict = _session_events.Data.from_dict

    def _from_dict(obj: Any):  # type: ignore[override]
        if isinstance(obj, dict):
            context = obj.get("context")
            if isinstance(context, (dict, list)):
                obj = dict(obj)
                obj["context"] = json.dumps(context, ensure_ascii=False)
        return original_from_dict(obj)

    _session_events.Data.from_dict = staticmethod(_from_dict)
    _session_events._context_patch_applied = True


_patch_copilot_context_parsing()


def _resolve_cli_path(configured_path: str | None) -> str | None:
    """Resolve Copilot CLI path to an absolute executable path when possible.

    The Copilot SDK validates cli_path using os.path.exists, so PATH-only values
    like "copilot" must be converted via shutil.which.
    """
    raw = (configured_path or "").strip()
    if raw:
        expanded = os.path.expanduser(raw)
        if any(sep in expanded for sep in ("/", "\\")):
            return expanded
        located = shutil.which(expanded)
        return located or expanded

    # No explicit path configured. If command is on PATH, use absolute path;
    # otherwise let SDK fall back to its bundled CLI binary.
    return shutil.which("copilot")


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


def _message_has_image(value: Any) -> bool:
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"image_url", "input_image", "image"}:
                return True
            if item_type == "tool_result" and _message_has_image(item.get("content")):
                return True
    return False


def _normalize_content_parts(value: Any, *, anthropic: bool = False) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return _coerce_text(value)

    parts: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            text = _coerce_text(item)
            if text:
                parts.append({"type": "text", "text": text})
            continue
        item_type = item.get("type")
        if item_type in {"text", "input_text"}:
            text = item.get("text") or item.get("content")
            if text:
                parts.append({"type": "text", "text": str(text)})
            continue
        if item_type in {"image_url", "input_image", "image"}:
            if anthropic:
                source = item.get("source")
                if isinstance(source, dict):
                    parts.append({"type": "image", "source": source})
                    continue
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                else:
                    url = image_url or item.get("url")
                if isinstance(url, str) and url.startswith("data:"):
                    media_type, _, data = url.partition(",")
                    media_type = media_type.removeprefix("data:").split(";", 1)[0] or "image/png"
                    parts.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
                elif isinstance(url, str):
                    parts.append({"type": "image", "source": {"type": "url", "url": url}})
                continue
            if item_type == "image_url":
                parts.append(dict(item))
                continue
            image_url = item.get("image_url") or item.get("url")
            if image_url:
                parts.append({"type": "image_url", "image_url": image_url if isinstance(image_url, dict) else {"url": image_url}})
            continue
    if not parts:
        return ""
    return parts


def _is_anthropic_messages_model(model: str) -> bool:
    model_id = (model or "").strip().lower()
    return model_id.startswith("claude-")


def _resolve_initiator(messages: Sequence[BaseMessage], configured: str | None) -> str:
    value = (configured or "").strip().lower()
    if value in {"user", "agent"}:
        return value
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return "user"
        if isinstance(msg, (AIMessage, ToolMessage, SystemMessage)):
            return "agent"
    return "user"


def _normalize_tool_result(result: Any) -> ToolResult:
    if result is None:
        return {"textResultForLlm": "", "resultType": "success"}
    if isinstance(result, dict) and "textResultForLlm" in result and "resultType" in result:
        return result  # type: ignore[return-value]
    if isinstance(result, str):
        return {"textResultForLlm": result, "resultType": "success"}
    try:
        json_str = json.dumps(result, default=str)
    except (TypeError, ValueError):
        json_str = str(result)
    return {"textResultForLlm": json_str, "resultType": "success"}


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    try:
        if hasattr(tool, "args_schema") and tool.args_schema:
            model_schema = tool.args_schema.model_json_schema()
            if isinstance(model_schema, dict):
                schema = model_schema
        elif hasattr(tool, "get_input_schema"):
            model = tool.get_input_schema()
            if hasattr(model, "model_json_schema"):
                model_schema = model.model_json_schema()
                if isinstance(model_schema, dict):
                    schema = model_schema
    except Exception:
        schema = {}
    return schema


async def _call_bound_callable(tool: Callable[..., Any], args: dict[str, Any]) -> Any:
    try:
        result = tool(**args)
    except TypeError:
        result = tool(args)
    if inspect.isawaitable(result):
        return await result
    return result


async def _invoke_langchain_tool(tool: Any, arguments: Any) -> ToolResult:
    args = arguments or {}
    result: Any
    if hasattr(tool, "ainvoke"):
        result = await tool.ainvoke(args)
    elif hasattr(tool, "invoke"):
        result = tool.invoke(args)
    elif hasattr(tool, "arun"):
        result = await tool.arun(args)
    elif hasattr(tool, "run"):
        result = tool.run(args)
    elif callable(tool):
        result = await _call_bound_callable(tool, args)
    else:
        result = f"Tool {getattr(tool, 'name', 'unknown')} is not callable."
    return _normalize_tool_result(result)


def _build_tool(tool: Any) -> Tool | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        description = tool.get("description") or ""
        handler = tool.get("handler")
        if not name or not callable(handler):
            return None

        async def _handler(invocation: dict[str, Any]) -> ToolResult:
            result = handler(invocation)
            if inspect.isawaitable(result):
                result = await result
            return _normalize_tool_result(result)

        return Tool(
            name=name,
            description=description,
            handler=_handler,
            parameters=tool.get("parameters"),
        )

    if hasattr(tool, "name") and hasattr(tool, "description"):
        name = tool.name
        description = tool.description or ""

        async def _handler(invocation: dict[str, Any]) -> ToolResult:
            return await _invoke_langchain_tool(tool, invocation.get("arguments"))

        return Tool(
            name=name,
            description=description,
            handler=_handler,
            parameters=_tool_schema(tool),
        )

    if callable(tool):
        name = getattr(tool, "__name__", "tool")
        description = getattr(tool, "__doc__", "") or ""

        async def _handler(invocation: dict[str, Any]) -> ToolResult:
            arguments = invocation.get("arguments") or {}
            result = await _call_bound_callable(tool, arguments)
            return _normalize_tool_result(result)

        return Tool(
            name=name,
            description=description,
            handler=_handler,
            parameters={},
        )

    return None


class ChatCopilot(BaseChatModel):
    """LangChain ChatModel wrapper for GitHub Copilot."""

    model: str = Field(default="gpt-5")
    temperature: float | None = Field(default=None)
    auth: CopilotAuth | None = Field(default=None, exclude=True)

    cli_path: str | None = Field(default=None)
    cli_url: str | None = Field(default=None)
    use_stdio: bool | None = Field(default=None)
    log_level: str | None = Field(default=None)
    session_timeout: float | None = Field(default=None, exclude=True)
    transport: str = Field(default="direct")
    api_base: str | None = Field(default=None)
    request_timeout: float = Field(default=120.0)
    initiator: str = Field(default="auto")
    max_tokens: int | None = Field(default=None)
    reasoning_effort: str | None = Field(default=None)
    auto_execute_tools: bool = Field(default=True)
    max_tool_rounds: int = Field(default=8)

    _tools: list[Tool] = PrivateAttr(default_factory=list)
    _openai_tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _anthropic_tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "copilot"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ChatCopilot":
        if tool_choice == "none":
            new_model = self.model_copy()
            new_model._tools = []
            new_model._openai_tools = []
            new_model._anthropic_tools = []
            return new_model

        copilot_tools: list[Tool] = []
        openai_tools: list[dict[str, Any]] = []
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            built = _build_tool(tool)
            if built:
                copilot_tools.append(built)
                schema = getattr(built, "parameters", None) or {}
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": getattr(built, "name", ""),
                            "description": getattr(built, "description", "") or "",
                            "parameters": schema,
                        },
                    }
                )
                anthropic_tools.append(
                    {
                        "name": getattr(built, "name", ""),
                        "description": getattr(built, "description", "") or "",
                        "input_schema": schema or {"type": "object", "properties": {}},
                    }
                )

        new_model = self.model_copy()
        new_model._tools = copilot_tools
        new_model._openai_tools = openai_tools
        new_model._anthropic_tools = anthropic_tools
        return new_model

    async def _ensure_auth(self) -> CopilotAuth | None:
        if self.auth is None:
            self.auth = load_copilot_auth_from_storage()
        if self.auth and self.auth.is_expired():
            raise ValueError("Copilot OAuth token expired - reauthenticate.")
        return self.auth

    def _build_prompt(self, messages: list[BaseMessage]) -> tuple[str, str | None]:
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages:
            content = _coerce_text(msg.content)
            if isinstance(msg, SystemMessage):
                if content:
                    system_parts.append(content)
            elif isinstance(msg, HumanMessage):
                conversation_parts.append(f"User: {content}")
            elif isinstance(msg, AIMessage):
                conversation_parts.append(f"Assistant: {content}")
            elif isinstance(msg, ToolMessage):
                conversation_parts.append(f"Tool: {content}")
            else:
                if content:
                    conversation_parts.append(content)

        system_message = "\n\n".join(system_parts).strip() if system_parts else None
        prompt = "\n\n".join(conversation_parts).strip()
        return prompt, system_message

    def _build_client(self, auth: CopilotAuth | None) -> CopilotClient:
        if CopilotClient is None:
            raise ImportError(
                "github-copilot-sdk is required for ChatCopilot transport='sdk'. "
                "Use transport='direct' or install github-copilot-sdk."
            )
        cli_url = self.cli_url or os.environ.get("COPILOT_CLI_URL")
        cli_path = _resolve_cli_path(self.cli_path or os.environ.get("COPILOT_CLI_PATH"))

        use_stdio = self.use_stdio
        if use_stdio is None:
            env_value = (os.environ.get("COPILOT_USE_STDIO") or "").strip().lower()
            if env_value:
                use_stdio = env_value in {"1", "true", "yes", "on"}

        log_level = self.log_level or os.environ.get("COPILOT_LOG_LEVEL") or "info"

        env = os.environ.copy()
        if auth and auth.access_token:
            env["GH_TOKEN"] = auth.access_token
            env["GITHUB_TOKEN"] = auth.access_token
        if auth and auth.enterprise_url:
            env["COPILOT_DOMAIN"] = auth.enterprise_url

        options: dict[str, Any] = {
            "log_level": log_level,
            "env": env,
        }
        if cli_url:
            options["cli_url"] = cli_url
        else:
            if cli_path:
                options["cli_path"] = cli_path
            if use_stdio is not None:
                options["use_stdio"] = use_stdio

        return CopilotClient(options)

    def _direct_base_url(self, auth: CopilotAuth | None) -> str:
        return (self.api_base or _copilot_api_base(auth.enterprise_url if auth else None)).rstrip("/")

    def _direct_headers(self, auth: CopilotAuth, messages: Sequence[BaseMessage]) -> dict[str, str]:
        if not auth.access_token:
            raise ValueError("Missing GitHub access token for Copilot")
        headers = {
            "Authorization": f"Bearer {auth.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "langchain-antigravity/copilot",
            "Openai-Intent": "conversation-edits",
            "x-initiator": _resolve_initiator(messages, self.initiator),
        }
        if any(_message_has_image(getattr(msg, "content", None)) for msg in messages):
            headers["Copilot-Vision-Request"] = "true"
        return headers

    def _convert_openai_messages(self, messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(getattr(msg, "tool_call_id", "") or ""),
                        "content": _coerce_text(msg.content),
                    }
                )
                continue
            else:
                role = "user"
            item: dict[str, Any] = {"role": role, "content": _normalize_content_parts(msg.content)}
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                tool_calls = []
                for call in msg.tool_calls:
                    tool_calls.append(
                        {
                            "id": call.get("id") or call.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": call.get("name"),
                                "arguments": json.dumps(call.get("args") or {}, default=str),
                            },
                        }
                    )
                item["tool_calls"] = tool_calls
            converted.append(item)
        return converted

    def _convert_anthropic_messages(self, messages: Sequence[BaseMessage]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                text = _coerce_text(msg.content)
                if text:
                    system_parts.append(text)
                continue
            if isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": _normalize_content_parts(msg.content, anthropic=True)})
            elif isinstance(msg, AIMessage):
                content = _normalize_content_parts(msg.content, anthropic=True)
                if getattr(msg, "tool_calls", None):
                    content_parts: list[dict[str, Any]] = []
                    if isinstance(content, str):
                        if content:
                            content_parts.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        content_parts.extend(content)
                    for call in msg.tool_calls:
                        content_parts.append(
                            {
                                "type": "tool_use",
                                "id": str(call.get("id") or call.get("call_id") or ""),
                                "name": str(call.get("name") or ""),
                                "input": call.get("args") if isinstance(call.get("args"), dict) else {},
                            }
                        )
                    content = content_parts
                converted.append({"role": "assistant", "content": content})
            elif isinstance(msg, ToolMessage):
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(getattr(msg, "tool_call_id", "") or ""),
                                "content": _coerce_text(msg.content),
                            }
                        ],
                    }
                )
        return ("\n\n".join(system_parts).strip() or None), converted

    def _parse_openai_message(self, payload: dict[str, Any]) -> AIMessage:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        tool_calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function") or {}
            args_raw = function.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else {}
            except Exception:
                args = {}
            tool_calls.append(
                ToolCall(
                    name=str(function.get("name") or ""),
                    args=args,
                    id=str(raw_call.get("id") or ""),
                )
            )
        metadata: dict[str, Any] = {}
        if message.get("reasoning_text"):
            metadata["reasoning"] = message.get("reasoning_text")
        if message.get("reasoning_opaque"):
            metadata["copilot_reasoning_opaque"] = message.get("reasoning_opaque")
        if payload.get("usage"):
            metadata["usage"] = payload.get("usage")
        kwargs: dict[str, Any] = {"content": content, "tool_calls": tool_calls}
        if metadata:
            kwargs["response_metadata"] = metadata
        return AIMessage(**kwargs)

    def _parse_anthropic_message(self, payload: dict[str, Any]) -> AIMessage:
        content_parts = payload.get("content") if isinstance(payload, dict) else None
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        metadata: dict[str, Any] = {}
        if isinstance(content_parts, list):
            for part in content_parts:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part_type in {"thinking", "reasoning"} and isinstance(part.get("thinking") or part.get("text"), str):
                    metadata["reasoning"] = part.get("thinking") or part.get("text")
                elif part_type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            name=str(part.get("name") or ""),
                            args=part.get("input") if isinstance(part.get("input"), dict) else {},
                            id=str(part.get("id") or ""),
                        )
                    )
        if payload.get("usage"):
            metadata["usage"] = payload.get("usage")
        kwargs: dict[str, Any] = {"content": "\n".join(text_parts), "tool_calls": tool_calls}
        if metadata:
            kwargs["response_metadata"] = metadata
        return AIMessage(**kwargs)

    def _tool_call_signature(self, tool_call: dict[str, Any]) -> str:
        name = str(tool_call.get("name") or "").strip()
        args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"

    def _tool_result_text(self, result: ToolResult) -> str:
        if isinstance(result, dict):
            text = result.get("textResultForLlm")
            if text is not None:
                return str(text)
            try:
                return json.dumps(result, default=str)
            except (TypeError, ValueError):
                return str(result)
        return str(result)

    async def _execute_direct_tool_calls(
        self,
        tool_calls: Sequence[dict[str, Any]],
    ) -> tuple[list[ToolMessage], list[dict[str, Any]]]:
        tools_by_name = {str(getattr(tool, "name", "") or "").strip(): tool for tool in self._tools}
        tool_messages: list[ToolMessage] = []
        handled: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name") or "").strip()
            call_id = str(tool_call.get("id") or tool_call.get("call_id") or "").strip()
            args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
            if not tool_name or not call_id:
                continue
            tool = tools_by_name.get(tool_name)
            if tool is None or not callable(getattr(tool, "handler", None)):
                raise RuntimeError(f"Copilot auto_execute_tools missing bound implementation for '{tool_name}'")

            success = True
            try:
                result = await tool.handler({"id": call_id, "name": tool_name, "arguments": args})
            except Exception as exc:
                success = False
                result = _normalize_tool_result({"error": str(exc)})

            result_text = self._tool_result_text(result)
            tool_messages.append(ToolMessage(content=result_text, tool_call_id=call_id))
            handled.append(
                {
                    "id": call_id,
                    "name": tool_name,
                    "args": args,
                    "output": result_text,
                    "success": success,
                }
            )
        return tool_messages, handled

    def _with_handled_tool_metadata(
        self,
        message: AIMessage,
        handled_tool_calls: list[dict[str, Any]],
    ) -> AIMessage:
        if not handled_tool_calls:
            return message
        metadata = dict(getattr(message, "response_metadata", {}) or {})
        metadata["handled_tool_calls"] = handled_tool_calls
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"response_metadata": metadata or None})
        return AIMessage(
            content=message.content,
            tool_calls=getattr(message, "tool_calls", None) or [],
            additional_kwargs=getattr(message, "additional_kwargs", None) or {},
            response_metadata=metadata or {},
        )

    def _message_from_stream_chunk(self, chunk: AIMessageChunk | None) -> AIMessage:
        if chunk is None:
            return AIMessage(content="")
        kwargs: dict[str, Any] = {
            "content": chunk.content or "",
            "tool_calls": getattr(chunk, "tool_calls", None) or [],
        }
        additional_kwargs = getattr(chunk, "additional_kwargs", None) or {}
        response_metadata = getattr(chunk, "response_metadata", None) or {}
        if additional_kwargs:
            kwargs["additional_kwargs"] = additional_kwargs
        if response_metadata:
            kwargs["response_metadata"] = response_metadata
        return AIMessage(**kwargs)

    async def _direct_context(self, messages: Sequence[BaseMessage]) -> tuple[str, dict[str, str]]:
        auth = await self._ensure_auth()
        if auth is None:
            raise ValueError("No Copilot authentication found. Run 'copilot-auth login' first or provide auth.")
        return self._direct_base_url(auth), self._direct_headers(auth, messages)

    def _guard_new_tool_calls(
        self,
        tool_calls: Sequence[dict[str, Any]],
        recent_tool_signatures: set[str],
    ) -> None:
        for tool_call in tool_calls:
            signature = self._tool_call_signature(tool_call)
            if signature in recent_tool_signatures:
                raise RuntimeError(
                    f"Copilot auto_execute_tools detected repeated tool call without assistant progress: {tool_call.get('name')}"
                )
            recent_tool_signatures.add(signature)

    def _build_anthropic_request_body(self, messages: Sequence[BaseMessage], *, stream: bool = False) -> dict[str, Any]:
        system_message, anthropic_messages = self._convert_anthropic_messages(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens or 4096,
        }
        if system_message:
            body["system"] = system_message
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self._anthropic_tools:
            body["tools"] = self._anthropic_tools
        if stream:
            body["stream"] = True
        return body

    def _build_openai_request_body(self, messages: Sequence[BaseMessage], *, stream: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_openai_messages(messages),
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self._openai_tools:
            body["tools"] = self._openai_tools
        if stream:
            body["stream"] = True
        return body

    async def _iter_sse_data(self, response: httpx.Response) -> AsyncIterator[str]:
        data_lines: list[str] = []
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                if data_lines:
                    pending = "\n".join(data_lines)
                    if pending == "[DONE]":
                        yield pending
                        data_lines = []
                    else:
                        try:
                            json.loads(pending)
                        except json.JSONDecodeError:
                            pass
                        else:
                            yield pending
                            data_lines = []
                data_lines.append(line[5:].lstrip())
        if data_lines:
            yield "\n".join(data_lines)

    async def _raise_for_stream_error(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        error_bytes = await response.aread()
        error_text = error_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(f"Copilot API error: {response.status_code} - {error_text}")

    async def _iter_sse_payloads(self, response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        async for data in self._iter_sse_data(response):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload

    def _openai_stream_chunk(self, payload: dict[str, Any]) -> AIMessageChunk | None:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            return None

        content = delta.get("content") or ""
        metadata: dict[str, Any] = {}
        for source_key, target_key in (
            ("reasoning_text", "reasoning"),
            ("reasoning_content", "reasoning"),
            ("reasoning", "reasoning"),
            ("reasoning_opaque", "copilot_reasoning_opaque"),
        ):
            value = delta.get(source_key)
            if value:
                metadata[target_key] = value
        if payload.get("usage"):
            metadata["usage"] = payload.get("usage")

        tool_call_chunks: list[ToolCallChunk] = []
        for raw_call in delta.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function") or {}
            tool_call_chunks.append(
                ToolCallChunk(
                    name=function.get("name"),
                    args=function.get("arguments") or "",
                    id=raw_call.get("id"),
                    index=raw_call.get("index"),
                )
            )
        if not content and not metadata and not tool_call_chunks:
            return None
        return AIMessageChunk(
            content=content,
            response_metadata=metadata,
            tool_call_chunks=tool_call_chunks,
        )

    def _anthropic_stream_chunk(self, payload: dict[str, Any]) -> tuple[AIMessageChunk | None, bool]:
        event_type = payload.get("type")
        if event_type == "message_stop":
            return None, True
        if event_type == "error":
            error = payload.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(str(message or "Copilot Anthropic stream error"))

        metadata: dict[str, Any] = {}
        tool_call_chunks: list[ToolCallChunk] = []
        content = ""
        index = payload.get("index")

        if event_type == "content_block_start":
            block = payload.get("content_block") or {}
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    content = block.get("text") or ""
                elif block_type in {"thinking", "reasoning"}:
                    reasoning = block.get("thinking") or block.get("text")
                    if reasoning:
                        metadata["reasoning"] = reasoning
                elif block_type == "tool_use":
                    tool_call_chunks.append(
                        ToolCallChunk(
                            name=block.get("name"),
                            args="",
                            id=block.get("id"),
                            index=index,
                        )
                    )
        elif event_type == "content_block_delta":
            delta = payload.get("delta") or {}
            if isinstance(delta, dict):
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    content = delta.get("text") or ""
                elif delta_type in {"thinking_delta", "reasoning_delta"}:
                    reasoning = delta.get("thinking") or delta.get("text")
                    if reasoning:
                        metadata["reasoning"] = reasoning
                elif delta_type == "input_json_delta":
                    tool_call_chunks.append(
                        ToolCallChunk(
                            name=None,
                            args=delta.get("partial_json") or "",
                            id=None,
                            index=index,
                        )
                    )
        elif event_type == "message_delta":
            delta = payload.get("delta") or {}
            if isinstance(delta, dict) and delta.get("stop_reason"):
                metadata["stop_reason"] = delta.get("stop_reason")
            if payload.get("usage"):
                metadata["usage"] = payload.get("usage")

        if not content and not metadata and not tool_call_chunks:
            return None, False
        return (
            AIMessageChunk(
                content=content,
                response_metadata=metadata,
                tool_call_chunks=tool_call_chunks,
            ),
            False,
        )

    async def _direct_astream_response(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        messages: list[BaseMessage],
    ) -> AsyncIterator[ChatGenerationChunk]:
        stream_headers = {**headers, "Accept": "text/event-stream"}
        if _is_anthropic_messages_model(self.model):
            response_context = client.stream(
                "POST",
                f"{base_url}/v1/messages",
                headers={**stream_headers, "anthropic-beta": "interleaved-thinking-2025-05-14"},
                json=self._build_anthropic_request_body(messages, stream=True),
            )
            async with response_context as response:
                await self._raise_for_stream_error(response)
                async for payload in self._iter_sse_payloads(response):
                    chunk, done = self._anthropic_stream_chunk(payload)
                    if chunk is not None:
                        yield ChatGenerationChunk(message=chunk)
                    if done:
                        break
            return

        response_context = client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers=stream_headers,
            json=self._build_openai_request_body(messages, stream=True),
        )
        async with response_context as response:
            await self._raise_for_stream_error(response)
            async for payload in self._iter_sse_payloads(response):
                chunk = self._openai_stream_chunk(payload)
                if chunk is not None:
                    yield ChatGenerationChunk(message=chunk)

    async def _direct_request_message(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict[str, str],
        messages: list[BaseMessage],
    ) -> AIMessage:
        if _is_anthropic_messages_model(self.model):
            body = self._build_anthropic_request_body(messages)
            response = await client.post(
                f"{base_url}/v1/messages",
                headers={**headers, "anthropic-beta": "interleaved-thinking-2025-05-14"},
                json=body,
            )
            if not response.is_success:
                raise RuntimeError(f"Copilot API error: {response.status_code} - {response.text}")
            return self._parse_anthropic_message(response.json())

        body = self._build_openai_request_body(messages)
        response = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
        if not response.is_success:
            raise RuntimeError(f"Copilot API error: {response.status_code} - {response.text}")
        return self._parse_openai_message(response.json())

    async def _direct_agenerate(self, messages: list[BaseMessage]) -> ChatResult:
        base_url, headers = await self._direct_context(messages)
        conversation = list(messages)
        handled_tool_calls: list[dict[str, Any]] = []
        recent_tool_signatures: set[str] = set()
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            for tool_round in range(self.max_tool_rounds + 1):
                message = await self._direct_request_message(client, base_url, headers, conversation)
                tool_calls = list(getattr(message, "tool_calls", None) or [])
                if not self.auto_execute_tools or not self._tools or not tool_calls:
                    message = self._with_handled_tool_metadata(message, handled_tool_calls)
                    llm_output = {"model_name": self.model, "transport": "direct"}
                    return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)

                self._guard_new_tool_calls(tool_calls, recent_tool_signatures)
                tool_messages, handled_batch = await self._execute_direct_tool_calls(tool_calls)
                handled_tool_calls.extend(handled_batch)
                conversation.append(message)
                conversation.extend(tool_messages)
                if tool_round == self.max_tool_rounds:
                    break

        raise RuntimeError(f"Copilot auto_execute_tools exceeded max_tool_rounds={self.max_tool_rounds}")

    async def _direct_astream(self, messages: list[BaseMessage]) -> AsyncIterator[ChatGenerationChunk]:
        base_url, headers = await self._direct_context(messages)
        conversation = list(messages)
        handled_tool_calls: list[dict[str, Any]] = []
        recent_tool_signatures: set[str] = set()
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            for tool_round in range(self.max_tool_rounds + 1):
                combined_chunk: AIMessageChunk | None = None
                async for generation_chunk in self._direct_astream_response(client, base_url, headers, conversation):
                    message_chunk = generation_chunk.message
                    combined_chunk = message_chunk if combined_chunk is None else combined_chunk + message_chunk
                    if self.auto_execute_tools and self._tools and getattr(message_chunk, "tool_call_chunks", None):
                        if not _coerce_text(message_chunk.content) and not getattr(message_chunk, "response_metadata", None):
                            continue
                    yield generation_chunk

                message = self._message_from_stream_chunk(combined_chunk)
                tool_calls = list(getattr(message, "tool_calls", None) or [])
                if not self.auto_execute_tools or not self._tools or not tool_calls:
                    if handled_tool_calls:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                response_metadata={"handled_tool_calls": handled_tool_calls},
                            )
                        )
                    return

                self._guard_new_tool_calls(tool_calls, recent_tool_signatures)
                tool_messages, handled_batch = await self._execute_direct_tool_calls(tool_calls)
                handled_tool_calls.extend(handled_batch)
                conversation.append(message)
                conversation.extend(tool_messages)
                if tool_round == self.max_tool_rounds:
                    break

        raise RuntimeError(f"Copilot auto_execute_tools exceeded max_tool_rounds={self.max_tool_rounds}")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if (self.transport or "direct").strip().lower() == "direct":
            return await self._direct_agenerate(messages)

        auth = await self._ensure_auth()
        prompt, system_message = self._build_prompt(messages)

        client = self._build_client(auth)
        session = None
        try:
            await client.start()
            session_config: dict[str, Any] = {"model": self.model}
            if system_message:
                session_config["system_message"] = {"mode": "append", "content": system_message}
            if self._tools:
                session_config["tools"] = self._tools

            session = await client.create_session(session_config)
            timeout = self.session_timeout
            if timeout is None:
                env_timeout = (os.environ.get("COPILOT_SESSION_TIMEOUT") or "").strip()
                if env_timeout:
                    try:
                        timeout = float(env_timeout)
                    except ValueError:
                        timeout = None
            if timeout is None:
                timeout = 180.0
            event = await session.send_and_wait({"prompt": prompt}, timeout=timeout)
            content = ""
            if event and hasattr(event.data, "content"):
                content = _coerce_text(event.data.content)
            message = AIMessage(content=content)
            llm_output = {"model_name": self.model}
            return ChatResult(generations=[ChatGeneration(message=message)], llm_output=llm_output)
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

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
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
        if (self.transport or "direct").strip().lower() == "direct":
            async for chunk in self._direct_astream(messages):
                token = _coerce_text(chunk.message.content)
                if token and run_manager is not None:
                    await run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
            return

        auth = await self._ensure_auth()
        prompt, system_message = self._build_prompt(messages)

        client = self._build_client(auth)
        session = None
        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _enqueue(event_type: str, payload: str | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))

        def _handler(event: SessionEvent) -> None:
            if event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                _enqueue("delta", _coerce_text(event.data.delta_content))
            elif event.type == SessionEventType.ASSISTANT_MESSAGE:
                _enqueue("final", _coerce_text(event.data.content))
            elif event.type == SessionEventType.SESSION_ERROR:
                _enqueue("error", getattr(event.data, "message", "Copilot session error"))
            elif event.type == SessionEventType.SESSION_IDLE:
                _enqueue("done", None)

        try:
            await client.start()
            session_config: dict[str, Any] = {"model": self.model, "streaming": True}
            if system_message:
                session_config["system_message"] = {"mode": "append", "content": system_message}
            if self._tools:
                session_config["tools"] = self._tools

            session = await client.create_session(session_config)
            session.on(_handler)
            await session.send({"prompt": prompt})

            while True:
                event_type, payload = await queue.get()
                if event_type == "delta" and payload:
                    chunk = AIMessageChunk(content=payload)
                    yield ChatGenerationChunk(message=chunk)
                elif event_type == "error":
                    raise RuntimeError(payload or "Copilot session error")
                elif event_type == "done":
                    break
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
