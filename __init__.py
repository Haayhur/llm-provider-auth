"""
LangChain Antigravity - Access Gemini 3/Claude via Google OAuth,
GPT-5.x/Codex via OpenAI OAuth, and GitHub Copilot via Copilot CLI SDK.

Based on:
- opencode-antigravity-auth by NoeFabris: https://github.com/NoeFabris/opencode-antigravity-auth
- opencode-openai-codex-auth by Numman Ali: https://github.com/numman-ali/opencode-openai-codex-auth
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Keep imports lightweight so CLI entrypoints like `ag-auth` don't fail if
# optional modules are absent in older/minimal installations.

from .auth import (
    AntigravityAuth,
    authorize_antigravity,
    exchange_token,
    refresh_access_token,
    load_accounts,
    save_accounts,
    interactive_login,
    load_auth_from_storage,
    list_accounts,
    set_active_account,
    remove_account,
)

if TYPE_CHECKING:  # pragma: no cover
    from .chat_model import ChatAntigravity as _ChatAntigravity
    from .codex_auth import CodexAuth as _CodexAuth
    from .codex_chat_model import ChatCodex as _ChatCodex
    from .copilot_auth import CopilotAuth as _CopilotAuth
    from .copilot_chat_model import ChatCopilot as _ChatCopilot


__all__ = [
    "AntigravityAuth",
    "authorize_antigravity",
    "exchange_token",
    "refresh_access_token",
    "load_accounts",
    "save_accounts",
    "interactive_login",
    "load_auth_from_storage",
    "list_accounts",
    "set_active_account",
    "remove_account",
]

# Optional Antigravity chat model export
try:  # pragma: no cover
    from .chat_model import ChatAntigravity

    __all__.insert(0, "ChatAntigravity")
except Exception:
    pass


# Optional Codex exports
try:  # pragma: no cover
    from .codex_chat_model import ChatCodex
    from .codex_auth import (
        CodexAuth,
        authorize_codex,
        exchange_codex_token,
        refresh_codex_token,
        load_codex_accounts,
        save_codex_accounts,
        codex_interactive_login,
        load_codex_auth_from_storage,
        list_codex_accounts,
        set_active_codex_account,
        remove_codex_account,
        normalize_codex_model,
    )

    __all__ += [
        "ChatCodex",
        "CodexAuth",
        "authorize_codex",
        "exchange_codex_token",
        "refresh_codex_token",
        "load_codex_accounts",
        "save_codex_accounts",
        "codex_interactive_login",
        "load_codex_auth_from_storage",
        "list_codex_accounts",
        "set_active_codex_account",
        "remove_codex_account",
        "normalize_codex_model",
    ]
except Exception:
    # Codex support not available in this installation.
    pass


# Optional Copilot exports
try:  # pragma: no cover
    from .copilot_chat_model import ChatCopilot
    from .copilot_auth import (
        CopilotAuth,
        request_copilot_device_code,
        exchange_copilot_device_code,
        load_copilot_accounts,
        save_copilot_accounts,
        copilot_interactive_login,
        load_copilot_auth_from_storage,
        list_copilot_accounts,
        set_active_copilot_account,
        remove_copilot_account,
        verify_copilot_token,
    )

    __all__ += [
        "ChatCopilot",
        "CopilotAuth",
        "request_copilot_device_code",
        "exchange_copilot_device_code",
        "load_copilot_accounts",
        "save_copilot_accounts",
        "copilot_interactive_login",
        "load_copilot_auth_from_storage",
        "list_copilot_accounts",
        "set_active_copilot_account",
        "remove_copilot_account",
        "verify_copilot_token",
    ]
except Exception:
    # Copilot support not available in this installation.
    pass


__version__ = "0.2.0"
