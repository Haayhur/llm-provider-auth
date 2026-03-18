# llm-provider-auth

Reusable OAuth auth + LangChain chat models for **Antigravity (Gemini/Claude)**, **OpenAI Codex (GPT-5.x)**, and **GitHub Copilot**.

This package provides native Python LangChain `ChatModel` implementations with OAuth authentication for both Google's Antigravity API and OpenAI's Codex backend, allowing you to use models like `gemini-3-flash`, `gpt-5.3-codex`, `claude-sonnet-4-6`, and more in your LangChain applications.

## Terms Warning (Antigravity)

> [!CAUTION]
> Using the Antigravity provider may violate Google's Terms of Service.
> Some users have reported account bans or shadow restrictions.
>
> Higher-risk cases reported by the reference plugin community:
> - Fresh Google accounts
> - New accounts with Pro/Ultra subscriptions
>
> By using Antigravity authentication here, you accept that risk.
> Prefer an established account that is not critical to your day-to-day services.

## Installation

From GitHub:

```bash
pip install "llm-provider-auth @ git+https://github.com/Haayhur/llm-provider-auth.git"
```

From local source:

```bash
pip install -e .
```

## Quick Start

### Option A: Google Antigravity (Gemini 3 & Claude models)

#### 1. Authenticate

```bash
ag-auth login
```

This opens your browser to sign in with Google. Credentials are stored locally.

#### 2. Use in LangChain

```python
from langchain_antigravity import ChatAntigravity

chat = ChatAntigravity(model="antigravity-gemini-3-flash")
response = chat.invoke("Hello! What's 2 + 2?")
print(response.content)
```

### Option B: OpenAI Codex (GPT-5.x & Codex models)

#### 1. Authenticate

```bash
codex-auth login
```

This opens your browser to sign in with ChatGPT/OpenAI. Credentials are stored locally.

#### 2. Use in LangChain

```python
from langchain_antigravity import ChatCodex

chat = ChatCodex(model="gpt-5.3-codex")
response = chat.invoke("Hello! What's 2 + 2?")
print(response.content)
```

### Option C: GitHub Copilot

#### 1. Authenticate

```bash
copilot-auth login
```

This opens your browser to sign in with GitHub. Credentials are stored locally.
Requires the GitHub Copilot CLI on PATH. You can either:
- Use device-code OAuth: run `copilot-auth login` (requires `COPILOT_CLIENT_ID`)
- Use a fine-grained PAT: set `GH_TOKEN` or `GITHUB_TOKEN` with "Copilot Requests" permission

#### 2. Use in LangChain

```python
from langchain_antigravity import ChatCopilot

chat = ChatCopilot(model="gpt-5")
response = chat.invoke("Hello! What's 2 + 2?")
print(response.content)
```

## Available Models

### Google Antigravity Models (via `ag-auth login`)

#### Gemini 3 Models

| Model | Description |
|-------|-------------|
| `antigravity-gemini-3-flash` | Fast, efficient model with thinking |
| `antigravity-gemini-3-pro-low` | Pro model with low thinking budget |
| `antigravity-gemini-3-pro-high` | Pro model with high thinking budget |

#### Claude Models

| Model | Description |
|-------|-------------|
| `antigravity-claude-sonnet-4-6` | Claude Sonnet 4.6 |
| `antigravity-claude-opus-4-6-thinking-low` | Opus 4.6 with low thinking budget |
| `antigravity-claude-opus-4-6-thinking-medium` | Opus 4.6 with medium thinking budget |
| `antigravity-claude-opus-4-6-thinking-high` | Opus 4.6 with high thinking budget |
| `antigravity-claude-opus-4-6-thinking-max` | Opus 4.6 with max thinking budget |

Backward-compatible `4.5` aliases are still accepted and mapped to the `4.6` lineup.

### OpenAI Codex Models (via `codex-auth login`)

#### Common models

| Model | Description |
|-------|-------------|
| `gpt-5.3-codex` | Latest GPT-5 Codex model |
| `gpt-5-codex` | GPT-5 Codex family model |
| `gpt-5.2` | GPT-5.2 base model |
| `gpt-5.2-codex` | GPT-5.2 Codex |
| `gpt-5.1-codex` | GPT-5.1 Codex |
| `gpt-5.1-codex-max` | GPT-5.1 Codex Max with frontend design focus |
| `gpt-5.1-codex-mini` | GPT-5.1 Codex Mini (medium/high) |
| `codex-mini-latest` | Alias for the latest Codex Mini |

Codex OAuth model policy in this package:
- Allowed: model IDs containing `codex`, plus `gpt-5.2`
- Rejected by validation: non-codex GPT-5 variants such as `gpt-5.1`, `gpt-5-pro`, `gpt-5-spark`

## Features

### Streaming

```python
from langchain_antigravity import ChatAntigravity

chat = ChatAntigravity(model="antigravity-gemini-3-flash")

for chunk in chat.stream("Count from 1 to 5"):
    print(chunk.content, end="", flush=True)
```

### Tool Calling

```python
from langchain_antigravity import ChatAntigravity

weather_tool = {
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
}

chat = ChatAntigravity(model="antigravity-gemini-3-flash").bind_tools([weather_tool])
response = chat.invoke("What's the weather in Paris?")

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc['name']}, Args: {tc['args']}")
```

### Multi-Account Support

```bash
# Add another account
ag-auth login

# List accounts
ag-auth accounts

# Switch active account
ag-auth accounts --set user@gmail.com

# Remove account
ag-auth logout user@gmail.com
```

## CLI Commands

### Google Antigravity Commands (`ag-auth`)

| Command | Description |
|---------|-------------|
| `ag-auth login` | Authenticate with Google |
| `ag-auth logout [email]` | Remove an account |
| `ag-auth accounts` | List authenticated accounts |
| `ag-auth accounts --set EMAIL` | Set active account |
| `ag-auth status` | Check authentication status |

### OpenAI Codex Commands (`codex-auth`)

| Command | Description |
|---------|-------------|
| `codex-auth login` | Authenticate with ChatGPT/OpenAI |
| `codex-auth logout [account-id]` | Remove an account |
| `codex-auth accounts` | List authenticated accounts |
| `codex-auth accounts --set ACCOUNT_ID` | Set active account |
| `codex-auth status` | Check authentication status |

## Configuration

### Google Antigravity Credentials
Stored in:
- **Windows**: `%APPDATA%\langchain-antigravity\accounts.json`
- **Linux/Mac**: `~/.config/langchain-antigravity/accounts.json`

### OpenAI Codex Credentials
Stored in:
- **Windows**: `%APPDATA%\langchain-antigravity\codex\accounts.json`
- **Linux/Mac**: `~/.config/langchain-antigravity/codex/accounts.json`

## API Reference

### ChatAntigravity (Google Antigravity)

```python
ChatAntigravity(
    model: str = "antigravity-gemini-3-flash",  # Model name
    temperature: float = None,                   # Sampling temperature
    max_output_tokens: int = None,               # Max output tokens
    auth: AntigravityAuth = None,                # Optional auth override
    project_id: str = None,                      # Optional project ID
)
```

### ChatCodex (OpenAI Codex)

```python
ChatCodex(
    model: str = "gpt-5.3-codex",             # Model name
    temperature: float = None,                 # Sampling temperature
    max_tokens: int = None,                    # Max output tokens
    reasoning_effort: str = None,              # Optional override (none/low/medium/high/xhigh)
    metadata: dict = None,                     # Optional Responses API metadata
    include: list[str] = None,                 # Optional Responses API include fields
    text: dict = None,                         # Optional Responses API text config
    truncation: str | dict = None,             # Optional Responses API truncation config
    parallel_tool_calls: bool = None,          # Optional Responses API parallel tool setting
    auto_execute_tools: bool = True,           # Auto-run bound function tools
    stream_structured_events: bool = False,    # Emit non-text stream events as chunks
    previous_response_id: str = None,          # Optional fixed Responses follow-up id
    computer_executor: callable = None,        # Optional host executor for computer_use_preview
    computer_loop_max_steps: int = 8,          # Max computer-use follow-up steps
    computer_loop_timeout_seconds: float = None,  # Optional timeout per computer-use step
    auth: CodexAuth = None,                    # Optional auth override
    account_id: str = None,                    # Optional account ID
    extra_body: dict = None,                   # Extra Responses API fields
)
```

Notes:
- If `reasoning_effort` is not provided, GPT-5 models default to `medium` with reasoning summary enabled.
- The Codex request validator enforces the OAuth model policy before API calls.
- Bound function tools are auto-executed by default. Set `auto_execute_tools=False` to get legacy `.tool_calls` behavior.
- Auto-executed function-tool traces are stored in `response_metadata["handled_tool_calls"]`.
- Auto-executed computer-use traces are stored in `response_metadata["handled_computer_calls"]`.
- Raw Responses output items are preserved in `additional_kwargs["codex_output_items"]` and `response_metadata["codex_output_items"]`.
- Prior assistant `response_id` values are reused automatically on the next turn unless you explicitly override `previous_response_id`.

### Methods

Both `ChatAntigravity` and `ChatCodex` support:
- `invoke(messages)` - Generate a response
- `stream(messages)` - Stream a response
- `bind_tools(tools)` - Bind tools for function calling

`ChatCodex` also supports:
- `bind_native_tools(tools)` - Bind provider-native Responses tools such as `{"type": "web_search"}`
- `bind_file_search_tools(...)` - Convenience helper for native `file_search`
- `bind_code_interpreter_tools(...)` - Convenience helper for native `code_interpreter`
- `bind_computer_use_tools(...)` - Convenience helper for native `computer_use_preview`
- `bind_mcp_tools(...)` - Convenience helper for native remote MCP tools
- `with_input_items(items)` - Attach exact Responses follow-up items to the next request

Example:

```python
def lookup_reference(*, query: str):
    return [{"title": f"hit:{query}"}]

chat = ChatCodex(model="gpt-5.3-codex").bind_tools([lookup_reference])
response = chat.invoke("Find polymer references")

print(response.content)
print(response.response_metadata["handled_tool_calls"])
```

Native hosted-tool example:

```python
from langchain_core.messages import HumanMessage

chat = (
    ChatCodex(model="gpt-5.3-codex")
    .bind_file_search_tools(vector_store_ids=["vs_123"], include_results=True)
    .bind_code_interpreter_tools(container={"type": "auto"}, reuse_previous_container=True)
)

response = chat.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "Find the most relevant paper and summarize it."},
                {"type": "image_url", "image_url": {"url": "https://example.com/figure.png", "detail": "high"}},
            ]
        )
    ]
)

print(response.response_metadata["response_id"])
print(response.response_metadata.get("file_search_results"))
print(response.response_metadata.get("code_interpreter"))
```

Computer-use example:

```python
def run_browser_step(computer_call: dict) -> dict:
    # Return a valid Responses computer_call_output payload.
    return {
        "output": {"image_url": "data:image/png;base64,..."},
        "current_url": "https://example.com",
    }

chat = ChatCodex(computer_executor=run_browser_step).bind_computer_use_tools(
    display_width=1440,
    display_height=900,
    environment="browser",
)
```

Native Responses notes:
- Vision inputs: pass OpenAI-style content blocks in `HumanMessage.content`, including `{"type": "image_url", ...}` or `{"type": "input_image", ...}`. `ChatCodex` normalizes them into Responses `input_image` items.
- Raw input escape hatch: attach `additional_kwargs["codex_input_items"]` to any message, or use `with_input_items(...)`, to append exact Responses input items such as approval responses or future tool payloads.
- `previous_response_id` is supported as a constructor field or per-call kwarg, and prior assistant `response_id` values are reused automatically on the next turn unless explicitly overridden.
- The resolved follow-up id is exposed as `response_metadata["previous_response_id_used"]`.
- File-search results are exposed in `response_metadata["file_search_results"]`.
- Code-interpreter metadata is exposed in `response_metadata["code_interpreter"]`; `bind_code_interpreter_tools(..., reuse_previous_container=True)` reuses the latest seen container id.
- Computer-use calls are exposed in `response_metadata["computer_calls"]`; if `computer_executor` is configured, ChatCodex will send `computer_call_output` items automatically until the turn completes.
- MCP calls are exposed in `response_metadata["mcp_calls"]`, and approval items in `response_metadata["mcp_approvals"]`. Use `build_mcp_approval_item(...)` from `langchain_antigravity.codex_responses` to construct approval responses safely.
- `with_input_items(...)` is model-level. Use it for reusable follow-up items; use per-message `codex_input_items` when the items belong to one specific turn.

Common response metadata:
- `reasoning` - flattened reasoning summary text
- `response_id` - current Responses id
- `previous_response_id_used` - follow-up id sent on this request
- `native_tool_events` - normalized native `*_call` items
- `file_search_results` - included file search result payloads
- `code_interpreter` - latest parsed container/file ids
- `computer_calls` - parsed computer-use calls
- `handled_tool_calls` - local function-tool executions performed by `ChatCodex`
- `handled_computer_calls` - local computer-use executions performed by `ChatCodex`
- `mcp_calls` / `mcp_approvals` - separated MCP call and approval items

### Codex Error Normalization

`ChatCodex` normalizes common backend error codes into clearer messages:
- `context_length_exceeded` → input exceeds model context window
- `insufficient_quota` → quota/billing message
- `usage_not_included` (and related plan text) → ChatGPT Plus upgrade guidance
- `invalid_prompt` → provider message (or generic fallback)

## Migration from OpenCode

If you previously used `opencode auth login`, this package can read those credentials. They're stored in the same format, just in a different location. To migrate:

```bash
# Your existing opencode credentials will be auto-detected
ag-auth status

# Or login fresh
ag-auth login
```

## Credits

This package is a Python port combining authentication and API logic from:

1. **opencode-antigravity-auth** by [@NoeFabris](https://github.com/NoeFabris) - Enables OpenCode to authenticate with Google's Antigravity API for Gemini 3 and Claude models.

2. **opencode-openai-codex-auth** by [@numman-ali](https://github.com/numman-ali) - Enables OpenCode to authenticate with OpenAI's ChatGPT backend for GPT-5.x and Codex models.

The original TypeScript plugins bring these functionalities to OpenCode. This Python package combines both and brings the same capabilities to LangChain applications.

## License

MIT

## Disclaimer

- This is an independent open-source project, not affiliated with Google
- "Antigravity", "Gemini", and "Google" are trademarks of Google LLC
- Use responsibly and in accordance with Google's terms of service
