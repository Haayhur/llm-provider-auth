"""Helpers for OpenAI Responses-style request/response handling."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Sequence


def _json_safe_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _tool_call_id(item: dict[str, Any]) -> str:
    value = item.get("call_id") or item.get("id") or ""
    return str(value or "").strip()


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").strip()
            if block_type in {"output_text", "text", "summary_text", "reasoning_text"}:
                text = block.get("text") or block.get("value") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)
    return ""


def parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return {}
    return {}


def serialize_function_call_output(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [_json_safe_copy(item) for item in value]
    return json.dumps(value, default=str)


def normalize_followup_input_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        item = _json_safe_copy(raw_item)
        item_type = str(item.get("type") or "").strip()
        if item_type in {"function_call_output", "custom_tool_call_output"} and "output" in item:
            item["output"] = serialize_function_call_output(item.get("output"))
        normalized.append(item)
    return normalized


def normalize_responses_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    normalized: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            normalized.append({"type": "input_text", "text": block})
            continue
        if not isinstance(block, dict):
            continue

        block_type = str(block.get("type") or "").strip()
        if block_type in {"input_text", "text", "output_text"}:
            text = block.get("text") or block.get("value") or ""
            normalized.append({"type": "input_text", "text": str(text)})
            continue
        if block_type in {"input_image", "image_url"}:
            image_url = block.get("image_url")
            file_id = block.get("file_id")
            detail = block.get("detail")
            image_block: dict[str, Any] = {"type": "input_image"}
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url:
                    image_block["image_url"] = url
                image_detail = image_url.get("detail")
                if isinstance(image_detail, str) and image_detail:
                    image_block["detail"] = image_detail
            elif isinstance(image_url, str) and image_url:
                image_block["image_url"] = image_url
            if isinstance(file_id, str) and file_id:
                image_block["file_id"] = file_id
            if isinstance(detail, str) and detail:
                image_block["detail"] = detail
            if "image_url" in image_block or "file_id" in image_block:
                normalized.append(image_block)
            continue
        normalized.append(_json_safe_copy(block))

    return normalized if normalized else content


def message_additional_input_items(message: Any) -> list[dict[str, Any]]:
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if not isinstance(additional_kwargs, dict):
        return []
    items = additional_kwargs.get("codex_input_items")
    if not isinstance(items, list):
        return []
    return [_json_safe_copy(item) for item in items if isinstance(item, dict)]


def build_mcp_approval_item(
    approval_request_id: str,
    approve: bool,
    *,
    reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "mcp_approval_response",
        "approval_request_id": approval_request_id,
        "approve": approve,
    }
    if reason is not None:
        item["reason"] = reason
    for key, value in extra.items():
        if value is not None:
            item[key] = _json_safe_copy(value)
    return item


def is_native_responses_tool(tool: Any) -> bool:
    return isinstance(tool, dict) and isinstance(tool.get("type"), str) and tool.get("type") != "function"


def split_bound_tools(
    tools: Sequence[dict[str, Any] | type | Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    function_tools: list[dict[str, Any]] = []
    native_tools: list[dict[str, Any]] = []

    for tool in tools:
        if is_native_responses_tool(tool):
            native_tools.append(_json_safe_copy(tool))
            continue
        if isinstance(tool, dict):
            if tool.get("type") == "function" and isinstance(tool.get("name"), str):
                function_tools.append(_json_safe_copy(tool))
            else:
                function_tools.append(
                    {
                        "type": "function",
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}) or {},
                    }
                )
            continue
        if isinstance(tool, type):
            function_tools.append(
                {
                    "type": "function",
                    "name": tool.__name__,
                    "description": tool.__doc__ or "",
                    "parameters": {},
                }
            )
            continue
        if callable(tool) and not isinstance(tool, type):
            function_tools.append(
                {
                    "type": "function",
                    "name": getattr(tool, "__name__", ""),
                    "description": getattr(tool, "__doc__", "") or "",
                    "parameters": {},
                }
            )
            continue
        if hasattr(tool, "name") and hasattr(tool, "description"):
            schema: Any = {}
            try:
                if hasattr(tool, "args_schema") and tool.args_schema:
                    schema_result = getattr(tool.args_schema, "model_json_schema", lambda: {})()
                    if isinstance(schema_result, dict):
                        schema = schema_result
            except Exception:
                pass
            function_tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                }
            )
    return function_tools, native_tools


def replayable_output_items(message: Any) -> list[dict[str, Any]]:
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        items = additional_kwargs.get("codex_output_items")
        if isinstance(items, list):
            return [_json_safe_copy(item) for item in items if isinstance(item, dict)]
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        items = response_metadata.get("codex_output_items")
        if isinstance(items, list):
            return [_json_safe_copy(item) for item in items if isinstance(item, dict)]
    return []


def latest_response_id(messages: Sequence[Any]) -> str | None:
    for message in reversed(messages):
        response_metadata = getattr(message, "response_metadata", None)
        if not isinstance(response_metadata, dict):
            continue
        response_id = str(response_metadata.get("response_id") or "").strip()
        if response_id:
            return response_id
    return None


def latest_code_interpreter_container_id(messages: Sequence[Any]) -> str | None:
    for message in reversed(messages):
        response_metadata = getattr(message, "response_metadata", None)
        if not isinstance(response_metadata, dict):
            continue
        code_interpreter = response_metadata.get("code_interpreter")
        if not isinstance(code_interpreter, dict):
            continue
        container_ids = code_interpreter.get("container_ids")
        if isinstance(container_ids, list):
            for value in reversed(container_ids):
                container_id = str(value or "").strip()
                if container_id:
                    return container_id
    return None


def normalize_responses_input_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()

    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        item = _json_safe_copy(raw_item)
        item_type = str(item.get("type") or "").strip()
        if item_type in {"function_call", "tool_call", "custom_tool_call"}:
            call_id = _tool_call_id(item)
            if not call_id:
                continue
            seen_call_ids.add(call_id)
            normalized.append(item)
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = _tool_call_id(item)
            if not call_id or call_id not in seen_call_ids:
                continue
            normalized.append(item)
            continue
        normalized.append(item)

    return normalized


@dataclass
class ParsedResponsesOutput:
    content: str
    reasoning: str | None
    external_tool_calls: list[dict[str, Any]]
    raw_output_items: list[dict[str, Any]]
    codex_items: list[dict[str, Any]]
    native_tool_events: list[dict[str, Any]]
    file_search_results: list[dict[str, Any]]
    code_interpreter: dict[str, Any] | None
    computer_calls: list[dict[str, Any]]
    mcp_calls: list[dict[str, Any]]
    mcp_approvals: list[dict[str, Any]]
    response_id: str | None


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = _json_safe_copy(item)
    if normalized.get("type") == "message":
        normalized["text"] = _extract_text(normalized.get("content"))
    return normalized


def _collect_values(value: Any, keys: set[str]) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {key: [] for key in keys}

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, nested in node.items():
                if key in keys and nested is not None:
                    if isinstance(nested, list):
                        found[key].extend(_json_safe_copy(nested))
                    else:
                        found[key].append(_json_safe_copy(nested))
                _visit(nested)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(value)
    return found


def parse_responses_output(response_data: dict[str, Any]) -> ParsedResponsesOutput:
    output = response_data.get("output", [])
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    external_tool_calls: list[dict[str, Any]] = []
    raw_output_items: list[dict[str, Any]] = []
    codex_items: list[dict[str, Any]] = []
    native_tool_events: list[dict[str, Any]] = []
    file_search_results: list[dict[str, Any]] = []
    code_interpreter_state: dict[str, Any] | None = None
    computer_calls: list[dict[str, Any]] = []
    mcp_calls: list[dict[str, Any]] = []
    mcp_approvals: list[dict[str, Any]] = []

    if isinstance(output, list):
        for raw_item in output:
            if not isinstance(raw_item, dict):
                continue
            item = _json_safe_copy(raw_item)
            raw_output_items.append(item)
            normalized = _normalize_item(item)
            codex_items.append(normalized)
            item_type = str(item.get("type") or "").strip()

            if item_type == "message":
                content = item.get("content")
                text = _extract_text(content)
                if text:
                    content_parts.append(text)
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = str(block.get("type") or "").strip()
                        if block_type in {"summary_text", "reasoning_text"}:
                            text = block.get("text") or block.get("value")
                            if isinstance(text, str) and text:
                                reasoning_parts.append(text)
                continue

            if item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, list):
                    for block in summary:
                        if not isinstance(block, dict):
                            continue
                        block_type = str(block.get("type") or "").strip()
                        if block_type in {"summary_text", "reasoning_text", "text"}:
                            text = block.get("text") or block.get("value")
                            if isinstance(text, str) and text:
                                reasoning_parts.append(text)
                else:
                    text = _extract_text(summary)
                    if text:
                        reasoning_parts.append(text)
                continue

            if item_type in {"function_call", "tool_call"}:
                external_tool_calls.append(
                    {
                        "name": str(item.get("name") or "").strip(),
                        "args": parse_arguments(item.get("arguments")),
                        "id": _tool_call_id(item),
                    }
                )
                continue

            if item_type.endswith("_call"):
                native_tool_events.append(normalized)
                if item_type == "file_search_call":
                    results = item.get("results")
                    if isinstance(results, list):
                        file_search_results.extend(
                            [_json_safe_copy(result) for result in results if isinstance(result, dict)]
                        )
                    continue
                if item_type == "code_interpreter_call":
                    fields = _collect_values(item, {"container_id", "file_id", "file_ids"})
                    code_interpreter_state = {
                        "item": normalized,
                        "container_ids": [str(value) for value in fields["container_id"] if str(value).strip()],
                        "file_ids": [
                            str(value)
                            for key in ("file_id", "file_ids")
                            for value in fields[key]
                            if str(value).strip()
                        ],
                    }
                    continue
                if item_type == "computer_call":
                    computer_calls.append(normalized)
                    continue
                if item_type.startswith("mcp_") or item_type == "mcp_call":
                    if "approval" in item_type:
                        mcp_approvals.append(normalized)
                    else:
                        mcp_calls.append(normalized)
                    continue

            if item_type.startswith("mcp_"):
                if "approval" in item_type:
                    mcp_approvals.append(normalized)
                else:
                    mcp_calls.append(normalized)

    top_level_reasoning = response_data.get("reasoning")
    if top_level_reasoning is not None and not reasoning_parts:
        if isinstance(top_level_reasoning, str):
            reasoning_parts.append(top_level_reasoning)
        else:
            reasoning_parts.append(json.dumps(top_level_reasoning, default=str))

    reasoning = "\n\n".join(part.strip() for part in reasoning_parts if part.strip()) or None
    return ParsedResponsesOutput(
        content="".join(content_parts),
        reasoning=reasoning,
        external_tool_calls=external_tool_calls,
        raw_output_items=raw_output_items,
        codex_items=codex_items,
        native_tool_events=native_tool_events,
        file_search_results=file_search_results,
        code_interpreter=code_interpreter_state,
        computer_calls=computer_calls,
        mcp_calls=mcp_calls,
        mcp_approvals=mcp_approvals,
        response_id=str(response_data.get("id") or "").strip() or None,
    )


@dataclass
class ResponsesSseAccumulator:
    output_items: list[dict[str, Any]] = field(default_factory=list)
    output_text_parts: list[str] = field(default_factory=list)
    reasoning_summary_parts: list[str] = field(default_factory=list)
    reasoning_text_parts: list[str] = field(default_factory=list)
    response_payload: dict[str, Any] | None = None
    final_event_type: str | None = None
    _seen_keys: set[str] = field(default_factory=set)

    def _record_item(self, item: Any) -> None:
        if not isinstance(item, dict):
            return
        stable_key = json.dumps(item, sort_keys=True, default=str)
        if stable_key in self._seen_keys:
            return
        self._seen_keys.add(stable_key)
        self.output_items.append(_json_safe_copy(item))

    def add_event(self, chunk_data: dict[str, Any]) -> None:
        event_type = str(chunk_data.get("type") or "").strip()
        if event_type in {"response.output_item.done", "response.output_item.added"}:
            self._record_item(chunk_data.get("item"))
            return
        if event_type in {"response.output_text.delta", "response.output_text.fragment"}:
            delta_text = chunk_data.get("delta") or chunk_data.get("text")
            if isinstance(delta_text, str) and delta_text:
                self.output_text_parts.append(delta_text)
            return
        if event_type == "response.output_item.delta":
            delta = chunk_data.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                text = _extract_text(content)
                if text:
                    self.output_text_parts.append(text)
            return
        if event_type == "response.reasoning_summary_text.delta":
            delta_text = chunk_data.get("delta") or chunk_data.get("text")
            if isinstance(delta_text, str) and delta_text:
                self.reasoning_summary_parts.append(delta_text)
            return
        if event_type == "response.reasoning_text.delta":
            delta_text = chunk_data.get("delta") or chunk_data.get("text")
            if isinstance(delta_text, str) and delta_text:
                self.reasoning_text_parts.append(delta_text)
            return
        if event_type in {"response.completed", "response.done", "response.failed", "response.incomplete"}:
            self.final_event_type = event_type
            response = chunk_data.get("response")
            if isinstance(response, dict):
                self.response_payload = _json_safe_copy(response)

    def build_response(self) -> dict[str, Any] | None:
        response: dict[str, Any] = _json_safe_copy(self.response_payload or {})
        output_items = [_json_safe_copy(item) for item in self.output_items]

        if not output_items and self.output_text_parts:
            output_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "".join(self.output_text_parts)}],
                }
            )
        if self.reasoning_summary_parts or self.reasoning_text_parts:
            if not any(str(item.get("type") or "") == "reasoning" for item in output_items):
                summary_blocks = [
                    {"type": "summary_text", "text": delta}
                    for delta in self.reasoning_summary_parts
                    if isinstance(delta, str) and delta
                ]
                if not summary_blocks and self.reasoning_text_parts:
                    summary_blocks = [
                        {"type": "reasoning_text", "text": delta}
                        for delta in self.reasoning_text_parts
                        if isinstance(delta, str) and delta
                    ]
                if summary_blocks:
                    output_items.insert(0, {"type": "reasoning", "summary": summary_blocks})

        if output_items:
            existing = response.get("output")
            if not isinstance(existing, list) or not existing:
                response["output"] = output_items
        if not response:
            return None
        return response
