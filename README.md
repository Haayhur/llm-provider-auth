# llm-provider-auth

Reusable OAuth auth + LangChain chat models for **Antigravity (Gemini/Claude)**, **OpenAI Codex (GPT-5.x)**, and **GitHub Copilot**.

This package provides native Python LangChain `ChatModel` implementations with OAuth authentication for both Google's Antigravity API and OpenAI's Codex backend, allowing you to use models like `gemini-3-flash`, `gpt-5.2-codex`, `claude-sonnet-4-5`, and more in your LangChain applications.

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

chat = ChatCodex(model="gpt-5.2-codex")
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
| `antigravity-claude-sonnet-4-5` | Claude Sonnet 4.5 |
| `antigravity-claude-sonnet-4-5-thinking-low` | Sonnet with 8K thinking budget |
| `antigravity-claude-sonnet-4-5-thinking-medium` | Sonnet with 16K thinking budget |
| `antigravity-claude-sonnet-4-5-thinking-high` | Sonnet with 32K thinking budget |
| `antigravity-claude-opus-4-5-thinking-low` | Opus with 8K thinking budget |
| `antigravity-claude-opus-4-5-thinking-medium` | Opus with 16K thinking budget |
| `antigravity-claude-opus-4-5-thinking-high` | Opus with 32K thinking budget |

### OpenAI Codex Models (via `codex-auth login`)

#### GPT-5.2 Models

| Model | Description |
|-------|-------------|
| `gpt-5.2` | GPT-5.2 general purpose (none/low/medium/high/xhigh) |
| `gpt-5.2-codex` | GPT-5.2 Codex with reasoning (low/medium/high/xhigh) |

#### GPT-5.1 Models

| Model | Description |
|-------|-------------|
| `gpt-5.1` | GPT-5.1 general purpose (none/low/medium/high) |
| `gpt-5.1-codex-max` | GPT-5.1 Codex Max with frontend design focus |
| `gpt-5.1-codex` | GPT-5.1 Codex (low/medium/high) |
| `gpt-5.1-codex-mini` | GPT-5.1 Codex Mini (medium/high) |

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
    model: str = "gpt-5.2-codex",             # Model name
    temperature: float = None,                       # Sampling temperature
    max_tokens: int = None,                         # Max output tokens
    reasoning_effort: str = None,                   # Reasoning effort: none/low/medium/high/xhigh
    auth: CodexAuth = None,                        # Optional auth override
    account_id: str = None,                          # Optional account ID
)
```

### Methods

Both `ChatAntigravity` and `ChatCodex` support:
- `invoke(messages)` - Generate a response
- `stream(messages)` - Stream a response
- `bind_tools(tools)` - Bind tools for function calling

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
