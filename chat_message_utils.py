"""Message conversion helpers for Antigravity chat models."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def convert_content(content: str | list[Any]) -> list[dict[str, Any]]:
    """Convert message content to parts."""
    if isinstance(content, str):
        return [{"text": content}]

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append({"text": item})
        elif isinstance(item, dict):
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                # Handle image content
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Base64 image
                    mime_type = url.split(";")[0].split(":")[1]
                    data = url.split(",")[1]
                    parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": data,
                        }
                    })

    return parts if parts else [{"text": ""}]


def convert_messages(messages: list[BaseMessage]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Convert LangChain messages to Antigravity format."""
    contents = []
    system_instruction = None

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_instruction = {
                "parts": [{"text": msg.content}]
            }
        elif isinstance(msg, HumanMessage):
            contents.append({
                "role": "user",
                "parts": convert_content(msg.content),
            })
        elif isinstance(msg, AIMessage):
            parts = []

            # Add text content
            if msg.content:
                parts.append({"text": msg.content})

            # Add tool calls
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                            "id": tool_call["id"],
                        }
                    })

            contents.append({
                "role": "model",
                "parts": parts,
            })
        elif isinstance(msg, ToolMessage):
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": msg.name or "unknown",
                        "response": {"result": msg.content},
                        "id": msg.tool_call_id,
                    }
                }]
            })

    return contents, system_instruction
