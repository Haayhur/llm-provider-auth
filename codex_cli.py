#!/usr/bin/env python3
"""
CLI for langchain-antigravity codex-auth.

Commands:
    login     - Authenticate with ChatGPT/OpenAI account
    logout    - Remove an account
    accounts  - List authenticated accounts
    status    - Check authentication status
"""

import argparse
import asyncio
import sys

from .codex_auth import (
    codex_interactive_login,
    load_codex_auth_from_storage,
    list_codex_accounts,
    set_active_codex_account,
    remove_codex_account,
    refresh_codex_token,
    get_codex_accounts_path,
)


def cmd_login(args):
    """Login with ChatGPT/OpenAI account."""
    async def do_login():
        await codex_interactive_login()

    try:
        asyncio.run(do_login())
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)


def cmd_logout(args):
    """Remove an account."""
    accounts = list_codex_accounts()
    if not accounts:
        print("No accounts found.")
        return

    if args.account_id:
        if remove_codex_account(args.account_id):
            print(f"Removed account: {args.account_id}")
        else:
            print(f"Account not found: {args.account_id}")
            sys.exit(1)
    else:
        print("Available accounts:")
        for i, acc in enumerate(accounts, 1):
            marker = "*" if acc["active"] else " "
            account_id = acc.get("account_id", "Unknown")
            print(f"  {marker} {i}. {account_id}")

        print("\nUse: codex-auth logout <account-id>")


def cmd_accounts(args):
    """List authenticated accounts."""
    accounts = list_codex_accounts()
    if not accounts:
        print("No accounts found. Run 'codex-auth login' to authenticate.")
        return

    print(f"Accounts ({get_codex_accounts_path()}):\n")
    for i, acc in enumerate(accounts, 1):
        marker = "(active)" if acc["active"] else ""
        account_id = acc.get("account_id", "Unknown")
        print(f"  {i}. {account_id} {marker}")

    if args.set:
        if set_active_codex_account(args.set):
            print(f"\nActive account set to: {args.set}")
        else:
            print(f"\nAccount not found: {args.set}")
            sys.exit(1)


def cmd_status(args):
    """Check authentication status."""
    auth = load_codex_auth_from_storage()
    if not auth:
        print("Not authenticated. Run 'codex-auth login' to authenticate.")
        sys.exit(1)

    print(f"Account ID: {auth.account_id}")

    async def check_token():
        try:
            refreshed = await refresh_codex_token(auth)
            print("Token:   Valid")
            return True
        except Exception as e:
            print(f"Token:   Invalid ({e})")
            return False

    if asyncio.run(check_token()):
        print("\nReady to use with ChatCodex!")
    else:
        print("\nRun 'codex-auth login' to re-authenticate.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="codex-auth",
        description="LangChain Antigravity - Access GPT-5.x and Codex models via ChatGPT OAuth",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    login_parser = subparsers.add_parser("login", help="Authenticate with ChatGPT/OpenAI account")
    login_parser.set_defaults(func=cmd_login)

    logout_parser = subparsers.add_parser("logout", help="Remove an account")
    logout_parser.add_argument("account_id", nargs="?", help="Account ID to remove")
    logout_parser.set_defaults(func=cmd_logout)

    accounts_parser = subparsers.add_parser("accounts", help="List authenticated accounts")
    accounts_parser.add_argument("--set", metavar="ACCOUNT_ID", help="Set active account")
    accounts_parser.set_defaults(func=cmd_accounts)

    status_parser = subparsers.add_parser("status", help="Check authentication status")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
