"""
GitHub Copilot device-code authentication helpers for LangChain.

Uses GitHub device-code OAuth to retrieve access tokens and account metadata.
Tokens can be used to authenticate the Copilot CLI via environment variables.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import webbrowser
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import httpx

try:
    from . import constants
except ImportError:  # pragma: no cover
    import constants  # type: ignore


OAUTH_POLLING_SAFETY_MARGIN_MS = 3000


@dataclass
class CopilotAuth:
    access_token: str
    expires_at: float
    account_id: str | None = None
    login: str | None = None
    email: str | None = None
    provider: str | None = None
    enterprise_url: str | None = None

    def is_expired(self, buffer_seconds: int = 60) -> bool:
        if not self.expires_at:
            return False
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "expires_at": self.expires_at,
            "account_id": self.account_id,
            "login": self.login,
            "email": self.email,
            "provider": self.provider,
            "enterprise_url": self.enterprise_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CopilotAuth":
        return cls(
            access_token=data.get("access_token", ""),
            expires_at=data.get("expires_at", 0),
            account_id=data.get("account_id"),
            login=data.get("login"),
            email=data.get("email"),
            provider=data.get("provider"),
            enterprise_url=data.get("enterprise_url") or data.get("enterpriseUrl"),
        )


@dataclass
class CopilotAccountStorage:
    version: int = 1
    accounts: list[dict[str, Any]] = field(default_factory=list)
    active_index: int = 0

    def get_active_account(self) -> dict[str, Any] | None:
        if 0 <= self.active_index < len(self.accounts):
            return self.accounts[self.active_index]
        return self.accounts[0] if self.accounts else None


def get_copilot_config_dir() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "langchain-antigravity" / "copilot"
    return Path.home() / ".config" / "langchain-antigravity" / "copilot"


def get_copilot_accounts_path() -> Path:
    return get_copilot_config_dir() / "accounts.json"


def load_copilot_accounts() -> CopilotAccountStorage | None:
    path = get_copilot_accounts_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CopilotAccountStorage(
            version=data.get("version", 1),
            accounts=data.get("accounts", []),
            active_index=data.get("activeIndex", 0),
        )
    except (json.JSONDecodeError, OSError):
        return None


def save_copilot_accounts(storage: CopilotAccountStorage) -> None:
    path = get_copilot_accounts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": storage.version,
        "accounts": storage.accounts,
        "activeIndex": storage.active_index,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_domain(url: str) -> str:
    cleaned = url.replace("https://", "").replace("http://", "").strip()
    cleaned = cleaned.split("/", 1)[0]
    return cleaned.rstrip("/")


def _resolve_copilot_domain(enterprise_url: str | None = None) -> str:
    if enterprise_url:
        return normalize_domain(enterprise_url)
    return "github.com"


def _is_enterprise_domain(domain: str) -> bool:
    return bool(domain) and domain != "github.com"


def _resolve_github_api_base(domain: str) -> str:
    if not domain or domain == "github.com":
        return constants.COPILOT_API_BASE
    return f"https://{domain}/api/v3"


def _get_device_urls(domain: str) -> dict[str, str]:
    return {
        "DEVICE_CODE_URL": f"https://{domain}/login/device/code",
        "ACCESS_TOKEN_URL": f"https://{domain}/login/oauth/access_token",
    }


async def request_copilot_device_code(
    *,
    scope: str | None = None,
    enterprise_url: str | None = None,
) -> dict[str, Any]:
    # OAuth device flow: request user_code + verification_uri, then poll for access_token.
    if not constants.COPILOT_CLIENT_ID:
        raise ValueError("COPILOT_CLIENT_ID is not configured")
    scope = scope or constants.COPILOT_SCOPE
    domain = _resolve_copilot_domain(enterprise_url)
    urls = _get_device_urls(domain)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                urls["DEVICE_CODE_URL"],
                json={"client_id": constants.COPILOT_CLIENT_ID, "scope": scope},
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        except httpx.RequestError as exc:
            raise ValueError(
                "Unable to reach GitHub device endpoint "
                f"at {urls['DEVICE_CODE_URL']}: {exc}"
            ) from exc

    if not response.is_success:
        raise ValueError(f"Failed to initiate device authorization: {response.text}")

    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = {}

    device_code = payload.get("device_code")
    user_code = payload.get("user_code")
    verification_uri = payload.get("verification_uri")
    interval = payload.get("interval") or 5

    if not device_code or not user_code or not verification_uri:
        raise ValueError("Invalid device authorization response")

    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "interval": interval,
        "domain": domain,
        "provider": "github-copilot-enterprise" if _is_enterprise_domain(domain) else "github-copilot",
        "enterprise_url": domain if _is_enterprise_domain(domain) else None,
    }


def _parse_token_payload(payload: dict[str, Any]) -> tuple[str, float]:
    access_token = payload.get("access_token", "")
    expires_in = payload.get("expires_in") or 0
    try:
        expires_in = float(expires_in)
    except (TypeError, ValueError):
        expires_in = 0
    return access_token, expires_in


def _pat_account_id(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"pat-{digest}"


async def _fetch_user_email(
    client: httpx.AsyncClient,
    access_token: str,
    api_base: str,
) -> str | None:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
    }
    resp = await client.get(f"{api_base}/user/emails", headers=headers)
    if not resp.is_success:
        return None
    try:
        emails = resp.json()
    except json.JSONDecodeError:
        return None
    if not isinstance(emails, list):
        return None
    primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
    candidate = primary or (emails[0] if emails else None)
    if isinstance(candidate, dict):
        value = candidate.get("email")
        if isinstance(value, str):
            return value
    return None


async def _fetch_account_profile(access_token: str, api_base: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{api_base}/user", headers=headers)
        if not resp.is_success:
            return {}
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            return {}
        email = payload.get("email")
        if not email:
            email = await _fetch_user_email(client, access_token, api_base)
        payload["email"] = email
        return payload


async def copilot_auth_from_pat(
    token: str,
    *,
    enterprise_url: str | None = None,
) -> CopilotAuth:
    token = (token or "").strip()
    if not token:
        raise ValueError("Missing PAT token")

    domain_value = _resolve_copilot_domain(enterprise_url)
    api_base = _resolve_github_api_base(domain_value)
    profile = await _fetch_account_profile(token, api_base)
    account_id = str(profile.get("id")) if profile.get("id") is not None else None
    login = profile.get("login") if isinstance(profile.get("login"), str) else None
    email = profile.get("email") if isinstance(profile.get("email"), str) else None
    if not account_id:
        account_id = login or email or _pat_account_id(token)
    provider = "github-copilot-enterprise" if _is_enterprise_domain(domain_value) else "github-copilot"
    enterprise_value = domain_value if _is_enterprise_domain(domain_value) else None
    return CopilotAuth(
        access_token=token,
        expires_at=0,
        account_id=account_id,
        login=login,
        email=email,
        provider=provider,
        enterprise_url=enterprise_value,
    )


async def exchange_copilot_device_code(
    device_code: str,
    *,
    interval: int = 5,
    enterprise_url: str | None = None,
    domain: str | None = None,
) -> CopilotAuth:
    """Poll the device-code token endpoint until an access token is issued."""
    if not constants.COPILOT_CLIENT_ID:
        raise ValueError("COPILOT_CLIENT_ID is not configured")
    if not device_code:
        raise ValueError("Missing device_code")

    domain_value = domain or _resolve_copilot_domain(enterprise_url)
    urls = _get_device_urls(domain_value)
    interval_seconds = max(1, int(interval or 5))

    start_time = time.time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                response = await client.post(
                    urls["ACCESS_TOKEN_URL"],
                    json={
                        "client_id": constants.COPILOT_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )
            except httpx.RequestError as exc:
                raise ValueError(
                    "Unable to reach GitHub token endpoint "
                    f"at {urls['ACCESS_TOKEN_URL']}: {exc}"
                ) from exc

            if not response.is_success:
                raise ValueError(f"Token polling failed: {response.text}")

            try:
                token_payload = response.json()
            except json.JSONDecodeError:
                token_payload = {}

            access_token = token_payload.get("access_token")
            if access_token:
                access_token, expires_in = _parse_token_payload(token_payload)
                api_base = _resolve_github_api_base(domain_value)
                profile = await _fetch_account_profile(access_token, api_base)
                account_id = str(profile.get("id")) if profile.get("id") is not None else None
                login = profile.get("login") if isinstance(profile.get("login"), str) else None
                email = profile.get("email") if isinstance(profile.get("email"), str) else None
                if not account_id:
                    account_id = login or email
                provider = "github-copilot-enterprise" if _is_enterprise_domain(domain_value) else "github-copilot"
                enterprise_value = domain_value if _is_enterprise_domain(domain_value) else None
                return CopilotAuth(
                    access_token=access_token,
                    expires_at=start_time + expires_in if expires_in else 0,
                    account_id=account_id,
                    login=login,
                    email=email,
                    provider=provider,
                    enterprise_url=enterprise_value,
                )

            error_code = token_payload.get("error")
            if error_code == "authorization_pending":
                await asyncio.sleep(interval_seconds + (OAUTH_POLLING_SAFETY_MARGIN_MS / 1000))
                continue
            if error_code == "slow_down":
                new_interval = interval_seconds + 5
                server_interval = token_payload.get("interval")
                if isinstance(server_interval, (int, float)) and server_interval > 0:
                    new_interval = int(server_interval)
                interval_seconds = max(1, new_interval)
                await asyncio.sleep(interval_seconds + (OAUTH_POLLING_SAFETY_MARGIN_MS / 1000))
                continue
            if error_code:
                raise ValueError(f"Device authorization failed: {error_code}")

            await asyncio.sleep(interval_seconds + (OAUTH_POLLING_SAFETY_MARGIN_MS / 1000))


def load_copilot_auth_from_storage() -> CopilotAuth | None:
    storage = load_copilot_accounts()
    if not storage or not storage.accounts:
        return None
    account = storage.get_active_account()
    if not account:
        return None
    return CopilotAuth(
        access_token=account.get("access_token", ""),
        expires_at=account.get("expires_at", 0),
        account_id=account.get("account_id"),
        login=account.get("login"),
        email=account.get("email"),
        provider=account.get("provider"),
        enterprise_url=account.get("enterprise_url") or account.get("enterpriseUrl"),
    )


def list_copilot_accounts() -> list[dict[str, Any]]:
    storage = load_copilot_accounts()
    if not storage:
        return []
    return [
        {
            "account_id": acc.get("account_id", "unknown"),
            "login": acc.get("login"),
            "email": acc.get("email"),
            "active": i == storage.active_index,
        }
        for i, acc in enumerate(storage.accounts)
    ]


def set_active_copilot_account(account_id: str) -> bool:
    storage = load_copilot_accounts()
    if not storage:
        return False
    for i, acc in enumerate(storage.accounts):
        if acc.get("account_id") == account_id:
            storage.active_index = i
            save_copilot_accounts(storage)
            return True
    return False


def remove_copilot_account(account_id: str) -> bool:
    storage = load_copilot_accounts()
    if not storage:
        return False
    for i, acc in enumerate(storage.accounts):
        if acc.get("account_id") == account_id:
            storage.accounts.pop(i)
            if storage.active_index >= len(storage.accounts):
                storage.active_index = max(0, len(storage.accounts) - 1)
            save_copilot_accounts(storage)
            return True
    return False


async def verify_copilot_token(auth: CopilotAuth) -> bool:
    if not auth.access_token:
        raise ValueError("Missing access token")
    api_base = _resolve_github_api_base(_resolve_copilot_domain(auth.enterprise_url))
    headers = {
        "Authorization": f"Bearer {auth.access_token}",
        "Accept": "application/vnd.github+json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{api_base}/user", headers=headers)
        if response.is_success:
            return True
        raise ValueError(f"Token validation failed: {response.status_code} {response.text}")
async def copilot_interactive_login(*, enterprise_url: str | None = None) -> CopilotAuth:
    device = await request_copilot_device_code(
        scope=constants.COPILOT_SCOPE,
        enterprise_url=enterprise_url,
    )

    verification_uri = device.get("verification_uri")
    user_code = device.get("user_code")
    print("\nOpen this URL in your browser to authenticate with GitHub Copilot:")
    print(f"{verification_uri}\n")
    print(f"Enter code: {user_code}")

    if verification_uri:
        webbrowser.open(verification_uri)

    auth = await exchange_copilot_device_code(
        device.get("device_code"),
        interval=int(device.get("interval") or 5),
        domain=device.get("domain"),
        enterprise_url=enterprise_url,
    )

    storage = load_copilot_accounts() or CopilotAccountStorage()
    account_id = auth.account_id or auth.login or auth.email or "unknown"

    existing_idx = next(
        (i for i, acc in enumerate(storage.accounts) if acc.get("account_id") == account_id),
        None,
    )

    payload = {
        "account_id": account_id,
        "access_token": auth.access_token,
        "expires_at": auth.expires_at,
        "login": auth.login,
        "email": auth.email,
        "provider": auth.provider,
        "enterprise_url": auth.enterprise_url,
        "addedAt": int(time.time() * 1000),
        "lastUsed": int(time.time() * 1000),
    }

    if existing_idx is not None:
        storage.accounts[existing_idx] = payload
        storage.active_index = existing_idx
    else:
        storage.accounts.append(payload)
        storage.active_index = len(storage.accounts) - 1

    save_copilot_accounts(storage)

    print(f"Authenticated (account: {auth.login or auth.email or account_id})")
    print(f"Credentials saved to {get_copilot_accounts_path()}")
    return auth
