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
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

try:
    from .copilot_auth import CopilotAuth, load_copilot_auth_from_storage
except Exception:  # pragma: no cover
    from copilot_auth import CopilotAuth, load_copilot_auth_from_storage  # type: ignore


try:
    from copilot import CopilotClient, Tool, ToolResult, SessionEvent  # type: ignore
    from copilot.generated.session_events import SessionEventType  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "github-copilot-sdk is required for Copilot provider. "
        "Install it with `pip install github-copilot-sdk`."
    ) from exc


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
        result = tool(args)
        if inspect.isawaitable(result):
            result = await result
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
            result = tool(invocation.get("arguments"))
            if inspect.isawaitable(result):
                result = await result
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

    _tools: list[Tool] = []

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
            return new_model

        copilot_tools: list[Tool] = []
        for tool in tools:
            built = _build_tool(tool)
            if built:
                copilot_tools.append(built)

        new_model = self.model_copy()
        new_model._tools = copilot_tools
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

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
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
