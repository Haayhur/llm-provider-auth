#!/usr/bin/env python3
"""
CLI for langchain-antigravity.

Commands:
    login     - Authenticate with Google account
    logout    - Remove an account
    accounts  - List authenticated accounts
    status    - Check authentication status
"""

import argparse
import asyncio
import sys

from .auth import (
    interactive_login,
    load_auth_from_storage,
    list_accounts,
    set_active_account,
    remove_account,
    refresh_access_token,
    get_accounts_path,
)


def cmd_login(args):
    """Login with Google account."""
    async def do_login():
        await interactive_login()
    
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
    accounts = list_accounts()
    if not accounts:
        print("No accounts found.")
        return
    
    if args.email:
        if remove_account(args.email):
            print(f"Removed account: {args.email}")
        else:
            print(f"Account not found: {args.email}")
            sys.exit(1)
    else:
        print("Available accounts:")
        for i, acc in enumerate(accounts, 1):
            marker = "*" if acc["active"] else " "
            print(f"  {marker} {i}. {acc['email']}")
        
        print("\nUse: ag-auth logout <email>")


def cmd_accounts(args):
    """List authenticated accounts."""
    accounts = list_accounts()
    if not accounts:
        print("No accounts found. Run 'ag-auth login' to authenticate.")
        return
    
    print(f"Accounts ({get_accounts_path()}):\n")
    for i, acc in enumerate(accounts, 1):
        marker = "(active)" if acc["active"] else ""
        print(f"  {i}. {acc['email']} {marker}")
    
    if args.set:
        if set_active_account(args.set):
            print(f"\nActive account set to: {args.set}")
        else:
            print(f"\nAccount not found: {args.set}")
            sys.exit(1)


def cmd_status(args):
    """Check authentication status."""
    auth = load_auth_from_storage()
    if not auth:
        print("Not authenticated. Run 'ag-auth login' to authenticate.")
        sys.exit(1)
    
    print(f"Account: {auth.email}")
    print(f"Project: {auth.project_id}")
    
    async def check_token():
        try:
            refreshed = await refresh_access_token(auth)
            print("Token:   Valid")
            return True
        except Exception as e:
            print(f"Token:   Invalid ({e})")
            return False
    
    if asyncio.run(check_token()):
        print("\nReady to use with ChatAntigravity!")
    else:
        print("\nRun 'ag-auth login' to re-authenticate.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="ag-auth",
        description="LangChain Antigravity - Access Gemini 3 and Claude models via Google OAuth",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    login_parser = subparsers.add_parser("login", help="Authenticate with Google account")
    login_parser.set_defaults(func=cmd_login)
    
    logout_parser = subparsers.add_parser("logout", help="Remove an account")
    logout_parser.add_argument("email", nargs="?", help="Email to remove")
    logout_parser.set_defaults(func=cmd_logout)
    
    accounts_parser = subparsers.add_parser("accounts", help="List authenticated accounts")
    accounts_parser.add_argument("--set", metavar="EMAIL", help="Set active account")
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
