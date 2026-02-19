"""
OAuth authentication for Antigravity API.

Based on opencode-antigravity-auth by NoeFabris
https://github.com/NoeFabris/opencode-antigravity-auth
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

try:
    from .constants import (
        ANTIGRAVITY_CLIENT_ID,
        ANTIGRAVITY_CLIENT_SECRET,
        ANTIGRAVITY_REDIRECT_URI,
        ANTIGRAVITY_SCOPES,
        ANTIGRAVITY_HEADERS,
        ANTIGRAVITY_LOAD_ENDPOINTS,
        ANTIGRAVITY_DEFAULT_PROJECT_ID,
    )
except ImportError:  # pragma: no cover
    from constants import (  # type: ignore
        ANTIGRAVITY_CLIENT_ID,
        ANTIGRAVITY_CLIENT_SECRET,
        ANTIGRAVITY_REDIRECT_URI,
        ANTIGRAVITY_SCOPES,
        ANTIGRAVITY_HEADERS,
        ANTIGRAVITY_LOAD_ENDPOINTS,
        ANTIGRAVITY_DEFAULT_PROJECT_ID,
    )


_LOGGER = logging.getLogger(__name__)
_AUTH_METRICS: dict[str, int] = {
    "antigravity_auth_refresh_success": 0,
    "antigravity_auth_refresh_revoked": 0,
    "antigravity_auth_refresh_failed": 0,
}


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


@dataclass
class AntigravityAuth:
    access_token: str
    refresh_token: str
    expires_at: float
    email: str | None = None
    project_id: str | None = None
    managed_project_id: str | None = None
    
    def is_expired(self, buffer_seconds: int = 60) -> bool:
        return time.time() >= (self.expires_at - buffer_seconds)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "email": self.email,
            "project_id": self.project_id,
            "managed_project_id": self.managed_project_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AntigravityAuth":
        return cls(
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            expires_at=data.get("expires_at", 0),
            email=data.get("email"),
            project_id=data.get("project_id"),
            managed_project_id=data.get("managed_project_id"),
        )


@dataclass
class AccountStorage:
    version: int = 3
    accounts: list[dict[str, Any]] = field(default_factory=list)
    active_index: int = 0
    
    def get_active_account(self) -> dict[str, Any] | None:
        if 0 <= self.active_index < len(self.accounts):
            return self.accounts[self.active_index]
        return self.accounts[0] if self.accounts else None


def _is_default_project_id(project_id: str | None) -> bool:
    return bool(project_id) and project_id == ANTIGRAVITY_DEFAULT_PROJECT_ID


def get_config_dir() -> Path:
    """Get the langchain-antigravity config directory."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "langchain-antigravity"
    return Path.home() / ".config" / "langchain-antigravity"


def get_legacy_config_dir() -> Path:
    """Get the legacy opencode config directory for migration."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "opencode"
    return Path.home() / ".config" / "opencode"


def get_accounts_path() -> Path:
    return get_config_dir() / "accounts.json"


def get_legacy_accounts_path() -> Path:
    return get_legacy_config_dir() / "antigravity-accounts.json"


def load_accounts() -> AccountStorage | None:
    path = get_accounts_path()
    
    if not path.exists():
        legacy_path = get_legacy_accounts_path()
        if legacy_path.exists():
            path = legacy_path
        else:
            return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AccountStorage(
            version=data.get("version", 3),
            accounts=data.get("accounts", []),
            active_index=data.get("activeIndex", 0),
        )
    except (json.JSONDecodeError, OSError):
        return None


def save_accounts(storage: AccountStorage) -> None:
    path = get_accounts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "version": storage.version,
        "accounts": storage.accounts,
        "activeIndex": storage.active_index,
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _upsert_account_from_auth(auth: AntigravityAuth) -> None:
    """Persist refreshed credentials for the matching account."""
    storage = load_accounts() or AccountStorage()
    now_ms = int(time.time() * 1000)

    existing_idx = None
    if auth.email:
        existing_idx = next(
            (i for i, acc in enumerate(storage.accounts) if acc.get("email") == auth.email),
            None,
        )

    if existing_idx is None:
        storage.accounts.append({
            "email": auth.email,
            "refreshToken": auth.refresh_token,
            "projectId": auth.project_id,
            "managedProjectId": auth.managed_project_id,
            "addedAt": now_ms,
            "lastUsed": now_ms,
        })
        storage.active_index = len(storage.accounts) - 1
    else:
        account = storage.accounts[existing_idx]
        account["email"] = auth.email or account.get("email")
        account["refreshToken"] = auth.refresh_token
        account["projectId"] = auth.project_id
        account["managedProjectId"] = auth.managed_project_id
        account["addedAt"] = account.get("addedAt", now_ms)
        account["lastUsed"] = now_ms
        storage.active_index = existing_idx

    save_accounts(storage)


def _clear_stored_refresh_token(auth: AntigravityAuth) -> None:
    """Clear stale refresh token after OAuth revocation."""
    storage = load_accounts()
    if not storage or not storage.accounts:
        return

    target_idx = None
    if auth.email:
        target_idx = next(
            (i for i, acc in enumerate(storage.accounts) if acc.get("email") == auth.email),
            None,
        )
    if target_idx is None and 0 <= storage.active_index < len(storage.accounts):
        target_idx = storage.active_index
    if target_idx is None:
        return

    account = storage.accounts[target_idx]
    account["refreshToken"] = ""
    account["lastUsed"] = int(time.time() * 1000)
    save_accounts(storage)


def _parse_oauth_error_payload(text: str | None) -> tuple[str, str]:
    if not text:
        return "", ""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "", text

    code = ""
    description = ""
    if isinstance(payload, dict):
        error_value = payload.get("error")
        if isinstance(error_value, str):
            code = error_value
        elif isinstance(error_value, dict):
            code = str(error_value.get("status") or error_value.get("code") or "").strip()
            description = str(error_value.get("message") or "").strip()

        description = description or str(payload.get("error_description") or payload.get("message") or "").strip()

    return code, description


def generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def encode_state(verifier: str, project_id: str = "", user_id: str = "") -> str:
    # Backwards compatible state payload. Older states only include verifier+projectId.
    payload = {"verifier": verifier, "projectId": project_id, "userId": user_id}
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def decode_state(state: str) -> tuple[str, str, str]:
    # urlsafe_b64encode strings may omit '=' padding; restore it safely.
    padded = state + "=" * (-len(state) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded))
        return (
            payload.get("verifier", ""),
            payload.get("projectId", ""),
            payload.get("userId", ""),
        )
    except (json.JSONDecodeError, ValueError):
        return "", "", ""


def authorize_antigravity(project_id: str = "", user_id: str = "") -> tuple[str, str]:
    verifier, challenge = generate_pkce()
    
    params = {
        "client_id": ANTIGRAVITY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
        "scope": " ".join(ANTIGRAVITY_SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": encode_state(verifier, project_id, user_id),
        "access_type": "offline",
        "prompt": "consent",
    }
    
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return url, verifier


async def fetch_project_id(access_token: str) -> str:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": ANTIGRAVITY_HEADERS["Client-Metadata"],
    }
    
    body = {
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint in ANTIGRAVITY_LOAD_ENDPOINTS:
            try:
                url = f"{endpoint}/v1internal:loadCodeAssist"
                response = await client.post(url, headers=headers, json=body)
                
                if not response.is_success:
                    continue
                
                data = response.json()
                project = data.get("cloudaicompanionProject")
                if isinstance(project, str) and project:
                    return project
                if isinstance(project, dict) and project.get("id"):
                    return project["id"]
            except Exception:
                continue
    
    return ""


async def exchange_token(
    code: str,
    state: str,
    redirect_uri: str | None = None,
) -> AntigravityAuth:
    verifier, project_id, _user_id = decode_state(state)
    
    if not verifier:
        raise ValueError("Missing PKCE verifier in state")
    
    start_time = time.time()
    
    redirect_value = redirect_uri or ANTIGRAVITY_REDIRECT_URI

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": ANTIGRAVITY_CLIENT_ID,
                "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_value,
                "code_verifier": verifier,
            },
        )
        
        if not response.is_success:
            raise ValueError(f"Token exchange failed: {response.text}")
        
        token_data = response.json()
        access_token = token_data.get("access_token", "")
        refresh_token = token_data.get("refresh_token", "")
        expires_in = token_data.get("expires_in", 3600)
        
        if not refresh_token:
            raise ValueError("Missing refresh token in response")
        
        email = None
        try:
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if user_response.is_success:
                email = user_response.json().get("email")
        except Exception:
            pass
        
        project_from_state = (project_id or "").strip()
        effective_project_id = project_from_state
        if not effective_project_id:
            effective_project_id = await fetch_project_id(access_token)
        if not project_from_state and _is_default_project_id(effective_project_id):
            effective_project_id = ""
        
        return AntigravityAuth(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=start_time + expires_in,
            email=email,
            project_id=effective_project_id or None,
        )


async def refresh_access_token(auth: AntigravityAuth) -> AntigravityAuth:
    if not auth.refresh_token:
        raise ValueError("Missing refresh token")
    
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": auth.refresh_token,
                "client_id": ANTIGRAVITY_CLIENT_ID,
                "client_secret": ANTIGRAVITY_CLIENT_SECRET,
            },
        )
        
        if not response.is_success:
            error_text = response.text
            error_code, error_description = _parse_oauth_error_payload(error_text)
            if error_code.lower() == "invalid_grant":
                _clear_stored_refresh_token(auth)
                _record_auth_metric(
                    "antigravity_auth_refresh_revoked",
                    email=auth.email,
                    project_id=auth.project_id,
                    code=error_code,
                )
                _LOGGER.warning(
                    "Antigravity refresh token revoked (email=%s, project_id=%s)",
                    auth.email,
                    auth.project_id,
                )
                hint = f" (email={auth.email})" if auth.email else ""
                raise ValueError(
                    f"Refresh token revoked{hint} - run 'antigravity login' again"
                )
            if error_code or error_description:
                _record_auth_metric(
                    "antigravity_auth_refresh_failed",
                    email=auth.email,
                    project_id=auth.project_id,
                    code=error_code,
                )
                details = ": ".join([part for part in (error_code, error_description) if part])
                raise ValueError(
                    f"Token refresh failed ({response.status_code}): {details}"
                )
            _record_auth_metric(
                "antigravity_auth_refresh_failed",
                email=auth.email,
                project_id=auth.project_id,
                code=error_code,
            )
            raise ValueError(f"Token refresh failed: {response.text}")
        
        token_data = response.json()
        
        access_token = token_data.get("access_token", "")

        # Some stored accounts may not have a usable projectId; refresh from token when missing.
        project_id = auth.project_id or ""
        if not project_id or _is_default_project_id(project_id):
            try:
                fetched_project = await fetch_project_id(access_token)
            except Exception:
                fetched_project = ""
            if fetched_project:
                project_id = fetched_project
            elif _is_default_project_id(project_id):
                project_id = ""

        refreshed = AntigravityAuth(
            access_token=access_token,
            refresh_token=token_data.get("refresh_token", auth.refresh_token),
            expires_at=start_time + token_data.get("expires_in", 3600),
            email=auth.email,
            project_id=project_id or None,
            managed_project_id=auth.managed_project_id,
        )
        _upsert_account_from_auth(refreshed)
        _record_auth_metric(
            "antigravity_auth_refresh_success",
            email=refreshed.email,
            project_id=refreshed.project_id,
        )
        return refreshed


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None
    state: str | None = None
    
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/oauth-callback":
            params = parse_qs(parsed.query)
            OAuthCallbackHandler.code = params.get("code", [None])[0]
            OAuthCallbackHandler.state = params.get("state", [None])[0]
            
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>Authentication Successful!</h1>
                <p>You can close this window and return to your terminal.</p>
                <script>window.close();</script>
                </body></html>
            """)
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format: str, *args: Any) -> None:
        pass


async def interactive_login(project_id: str = "") -> AntigravityAuth:
    OAuthCallbackHandler.code = None
    OAuthCallbackHandler.state = None
    
    auth_url, verifier = authorize_antigravity(project_id)
    
    server = HTTPServer(("localhost", 51121), OAuthCallbackHandler)
    server_thread = Thread(target=server.handle_request, daemon=True)
    server_thread.start()
    
    print("\nOpening browser for Google authentication...")
    print(f"If browser doesn't open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)
    
    server_thread.join(timeout=120)
    server.server_close()
    
    if not OAuthCallbackHandler.code or not OAuthCallbackHandler.state:
        raise ValueError("OAuth callback not received - authentication timed out")
    
    auth = await exchange_token(OAuthCallbackHandler.code, OAuthCallbackHandler.state)
    
    storage = load_accounts() or AccountStorage()
    
    existing_idx = next(
        (i for i, acc in enumerate(storage.accounts) if acc.get("email") == auth.email),
        None
    )
    
    if existing_idx is not None:
        storage.accounts[existing_idx] = {
            "email": auth.email,
            "refreshToken": auth.refresh_token,
            "projectId": auth.project_id,
            "managedProjectId": auth.managed_project_id,
            "addedAt": storage.accounts[existing_idx].get("addedAt", int(time.time() * 1000)),
            "lastUsed": int(time.time() * 1000),
        }
        storage.active_index = existing_idx
    else:
        storage.accounts.append({
            "email": auth.email,
            "refreshToken": auth.refresh_token,
            "projectId": auth.project_id,
            "managedProjectId": auth.managed_project_id,
            "addedAt": int(time.time() * 1000),
            "lastUsed": int(time.time() * 1000),
        })
        storage.active_index = len(storage.accounts) - 1
    
    save_accounts(storage)
    
    print(f"Authenticated as {auth.email}")
    print(f"Credentials saved to {get_accounts_path()}")
    return auth


def load_auth_from_storage() -> AntigravityAuth | None:
    storage = load_accounts()
    if not storage or not storage.accounts:
        return None
    
    account = storage.get_active_account()
    if not account:
        return None
    
    return AntigravityAuth(
        access_token="",
        refresh_token=account.get("refreshToken", ""),
        expires_at=0,
        email=account.get("email"),
        project_id=account.get("projectId"),
        managed_project_id=account.get("managedProjectId"),
    )


def list_accounts() -> list[dict[str, Any]]:
    storage = load_accounts()
    if not storage:
        return []
    return [
        {"email": acc.get("email"), "active": i == storage.active_index}
        for i, acc in enumerate(storage.accounts)
    ]


def set_active_account(email: str) -> bool:
    storage = load_accounts()
    if not storage:
        return False
    
    for i, acc in enumerate(storage.accounts):
        if acc.get("email") == email:
            storage.active_index = i
            save_accounts(storage)
            return True
    return False


def remove_account(email: str) -> bool:
    storage = load_accounts()
    if not storage:
        return False
    
    for i, acc in enumerate(storage.accounts):
        if acc.get("email") == email:
            storage.accounts.pop(i)
            if storage.active_index >= len(storage.accounts):
                storage.active_index = max(0, len(storage.accounts) - 1)
            save_accounts(storage)
            return True
    return False
