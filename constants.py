"""
Constants for Antigravity OAuth and API integration.

Based on opencode-antigravity-auth by NoeFabris:
https://github.com/NoeFabris/opencode-antigravity-auth
"""

import importlib
import os

# OAuth credentials: env-only for safe OSS distribution.
# Configure ANTIGRAVITY_CLIENT_ID and ANTIGRAVITY_CLIENT_SECRET at runtime.
ANTIGRAVITY_CLIENT_ID = os.environ.get("ANTIGRAVITY_CLIENT_ID", "").strip()
ANTIGRAVITY_CLIENT_SECRET = os.environ.get("ANTIGRAVITY_CLIENT_SECRET", "").strip()

ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

_render_external_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip()
if _render_external_url:
    _render_external_url = _render_external_url.rstrip("/")
_antigravity_default_redirect = (
    f"{_render_external_url}/api/antigravity/callback"
    if _render_external_url
    else "http://localhost:51121/oauth-callback"
)
ANTIGRAVITY_REDIRECT_URI = os.environ.get(
    "ANTIGRAVITY_REDIRECT_URI",
    _antigravity_default_redirect,
)

ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_AUTOPUSH = "https://autopush-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"

ANTIGRAVITY_ENDPOINT_FALLBACKS = [
    # Prefer prod first; sandbox endpoints can return 403 for some accounts/projects.
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
]

ANTIGRAVITY_LOAD_ENDPOINTS = [
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
]

ANTIGRAVITY_ENDPOINT = ANTIGRAVITY_ENDPOINT_DAILY
GEMINI_CLI_ENDPOINT = ANTIGRAVITY_ENDPOINT_PROD

ANTIGRAVITY_DEFAULT_PROJECT_ID = "rising-fact-p41fc"
ANTIGRAVITY_PROJECT_ID = os.environ.get("ANTIGRAVITY_PROJECT_ID", "").strip() or None

ANTIGRAVITY_HEADERS = {
    "User-Agent": "antigravity/1.11.5 windows/amd64",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}

GEMINI_CLI_HEADERS = {
    "User-Agent": "google-api-nodejs-client/9.15.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
}

DEFAULT_THINKING_BUDGET = 16000

EMPTY_SCHEMA_PLACEHOLDER_NAME = "_placeholder"
EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION = "Placeholder. Always pass true."

CLAUDE_TOOL_SYSTEM_INSTRUCTION = """CRITICAL TOOL USAGE INSTRUCTIONS:
You are operating in a custom environment where tool definitions differ from your training data.
You MUST follow these rules strictly:

1. DO NOT use your internal training data to guess tool parameters
2. ONLY use the exact parameter structure defined in the tool schema
3. Parameter names in schemas are EXACT - do not substitute with similar names from your training
4. Array parameters have specific item types - check the schema's 'items' field for the exact structure
5. When you see "STRICT PARAMETERS" in a tool description, those type definitions override any assumptions
6. Tool use in agentic workflows is REQUIRED - you must call tools with the exact parameters specified

If you are unsure about a tool's parameters, YOU MUST read the schema definition carefully."""

MODEL_MAPPINGS = {
    "antigravity-gemini-3-flash": "gemini-3-flash",
    "antigravity-gemini-3-pro-low": "gemini-3-pro-low",
    "antigravity-gemini-3-pro-high": "gemini-3-pro-high",
    "antigravity-claude-sonnet-4-5": "claude-sonnet-4-5",
    "antigravity-claude-sonnet-4-5-thinking-low": "claude-sonnet-4-5-thinking",
    "antigravity-claude-sonnet-4-5-thinking-medium": "claude-sonnet-4-5-thinking",
    "antigravity-claude-sonnet-4-5-thinking-high": "claude-sonnet-4-5-thinking",
    "antigravity-claude-opus-4-5-thinking-low": "claude-opus-4-5-thinking",
    "antigravity-claude-opus-4-5-thinking-medium": "claude-opus-4-5-thinking",
    "antigravity-claude-opus-4-5-thinking-high": "claude-opus-4-5-thinking",
    "gemini-2.5-flash": "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
    "gemini-3-flash-preview": "gemini-3.0-flash-preview",
    "gemini-3-pro-preview": "gemini-3.0-pro-preview",
}

THINKING_BUDGETS = {
    "low": 8000,
    "medium": 16000,
    "high": 32000,
}

GEMINI_THINKING_LEVELS = {
    "flash": "minimal",
    "low": "low",
    "high": "high",
}

# OpenAI Codex OAuth Constants
CODEX_CLIENT_ID = os.environ.get(
    "CODEX_CLIENT_ID",
    "app_EMoamEEZ73f0CkXaXp7hrann"
)
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_REDIRECT_URI = os.environ.get(
    "CODEX_REDIRECT_URI",
    "http://localhost:1455/auth/callback"
)

def get_codex_redirect_uri() -> str:
    """Get the OAuth redirect URI, adjusting for Docker environments.
    
    In Docker, the dashboard runs on a different host/port than the callback.
    If CODEX_REDIRECT_URI is explicitly set via env var, use that.
    Otherwise, detect Docker and adjust to use the actual API host.
    """
    # If explicitly set, respect it
    redirect_from_env = os.environ.get("CODEX_REDIRECT_URI")
    if redirect_from_env:
        return redirect_from_env

    render_external_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip()
    if render_external_url:
        render_external_url = render_external_url.rstrip("/")
        return f"{render_external_url}/api/codex/callback"
    
    # Check if running in Docker (common indicators)
    in_docker = (
        os.path.exists("/.dockerenv") or
        os.getenv("KUBERNETES_SERVICE_HOST") or
        os.getenv("CONTAINER_NAME") or
        os.getenv("DOCKER_CONTAINER")
    )
    
    if in_docker:
        # In Docker, use the API host from environment
        api_host = os.getenv("API_HOST", "localhost:8000")
        api_host = api_host.replace(":8000", ":1455")  # Codex uses port 1455
        return f"http://{api_host}/auth/callback"
    
    # Default local development
    return "http://localhost:1455/auth/callback"

CODEX_SCOPE = "openid profile email offline_access"
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_DUMMY_API_KEY = "chatgpt-oauth"

CODEX_HEADERS = {
    "OpenAI-Beta": "responses=experimental",
    "originator": "codex_cli_rs",
}

# ChatGPT Codex backend requires an `instructions` string in the request body.
# The official Codex CLI fetches model-specific instructions; we provide a small
# built-in default to keep the integration functional without extra network calls.
CODEX_DEFAULT_INSTRUCTIONS = "You are Codex, a helpful coding assistant."

CODEX_MODEL_MAPPINGS = {
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-none": "gpt-5.2",
    "gpt-5.2-low": "gpt-5.2",
    "gpt-5.2-medium": "gpt-5.2",
    "gpt-5.2-high": "gpt-5.2",
    "gpt-5.2-xhigh": "gpt-5.2",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-5.2-codex-low": "gpt-5.2-codex",
    "gpt-5.2-codex-medium": "gpt-5.2-codex",
    "gpt-5.2-codex-high": "gpt-5.2-codex",
    "gpt-5.2-codex-xhigh": "gpt-5.2-codex",
    "gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "gpt-5.1-codex-max-low": "gpt-5.1-codex-max",
    "gpt-5.1-codex-max-medium": "gpt-5.1-codex-max",
    "gpt-5.1-codex-max-high": "gpt-5.1-codex-max",
    "gpt-5.1-codex-max-xhigh": "gpt-5.1-codex-max",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1-codex-low": "gpt-5.1-codex",
    "gpt-5.1-codex-medium": "gpt-5.1-codex",
    "gpt-5.1-codex-high": "gpt-5.1-codex",
    "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    "gpt-5.1-codex-mini-medium": "gpt-5.1-codex-mini",
    "gpt-5.1-codex-mini-high": "gpt-5.1-codex-mini",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.1-none": "gpt-5.1",
    "gpt-5.1-low": "gpt-5.1",
    "gpt-5.1-medium": "gpt-5.1",
    "gpt-5.1-high": "gpt-5.1",
    "gpt-5-codex": "gpt-5.1-codex",
    "gpt-5-codex-low": "gpt-5.1-codex",
    "gpt-5-codex-medium": "gpt-5.1-codex",
    "gpt-5-codex-high": "gpt-5.1-codex",
    "gpt-5-codex-mini": "gpt-5.1-codex-mini",
    "gpt-5": "gpt-5.1",
    "gpt-5-none": "gpt-5.1",
    "gpt-5-low": "gpt-5.1",
    "gpt-5-medium": "gpt-5.1",
    "gpt-5-high": "gpt-5.1",
    "gpt-5-mini": "gpt-5.1",
    "gpt-5-nano": "gpt-5.1",
    "codex-mini-latest": "gpt-5.1-codex-mini",
}

# Antigravity model prefixes for identification
ANTIGRAVITY_MODEL_PREFIX = "antigravity-"
CODEX_MODEL_PREFIX = "openai/"

def _resolve_copilot_client_id() -> str:
    explicit = os.environ.get("COPILOT_CLIENT_ID", "").strip()
    if explicit:
        return explicit
    for key in ("GITHUB_COPILOT_CLIENT_ID", "GH_COPILOT_CLIENT_ID"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    for module_name in ("copilot", "copilot.constants"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr in (
            "CLIENT_ID",
            "DEFAULT_CLIENT_ID",
            "DEFAULT_COPILOT_CLIENT_ID",
            "GITHUB_CLIENT_ID",
            "COPILOT_CLIENT_ID",
        ):
            value = getattr(module, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for attr, value in vars(module).items():
            if "CLIENT_ID" in attr and isinstance(value, str) and value.strip():
                return value.strip()
    return "Ov23li8tweQw6odWQebz"


# GitHub Copilot OAuth Constants
COPILOT_CLIENT_ID = _resolve_copilot_client_id()
COPILOT_CLIENT_SECRET = os.environ.get("COPILOT_CLIENT_SECRET", "")
COPILOT_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
COPILOT_TOKEN_URL = "https://github.com/login/oauth/access_token"
COPILOT_SCOPE = os.environ.get("COPILOT_SCOPE", "read:user user:email")
COPILOT_REDIRECT_URI = os.environ.get(
    "COPILOT_REDIRECT_URI",
    "http://localhost:8000/api/copilot/callback",
)
COPILOT_API_BASE = "https://api.github.com"
COPILOT_DOMAIN = os.environ.get("COPILOT_DOMAIN", "").strip()
COPILOT_ENTERPRISE_URL = os.environ.get("COPILOT_ENTERPRISE_URL", "").strip()
