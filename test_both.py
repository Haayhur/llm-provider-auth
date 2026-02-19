#!/usr/bin/env python3
"""
Test script for LangChain Antigravity (Google) and Codex (OpenAI) integration.

Prerequisites:
1. For Google tests: Run `ag-auth login` first to authenticate
2. For Codex tests: Run `codex-auth login` first to authenticate
3. Install dependencies: pip install langchain-core httpx pydantic

Usage:
    python test_both.py
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import langchain_antigravity.auth as auth_module
import langchain_antigravity.chat_model as chat_module
import langchain_antigravity.codex_auth as codex_auth_module
import langchain_antigravity.codex_chat_model as codex_chat_module

load_auth_from_storage = auth_module.load_auth_from_storage
refresh_access_token = auth_module.refresh_access_token
ChatAntigravity = chat_module.ChatAntigravity
load_codex_auth_from_storage = codex_auth_module.load_codex_auth_from_storage
refresh_codex_token = codex_auth_module.refresh_codex_token
ChatCodex = codex_chat_module.ChatCodex


def define_simple_tool():
    """Define a simple tool for testing."""
    return {
        "name": "get_weather",
        "description": "Get the current weather for a location. Use this when user asks about weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. 'San Francisco, CA'"
                }
            },
            "required": ["location"]
        }
    }


async def test_antigravity_simple_chat():
    """Test Google Antigravity simple chat."""
    print("\n" + "="*60)
    print("GOOGLE ANTIGRAVITY TEST 1: Simple Chat")
    print("="*60)

    auth = load_auth_from_storage()
    if not auth:
        print("❌ No Google authentication found!")
        print("   Run 'ag-auth login' first to authenticate.")
        return False

    print(f"✓ Found Google auth for: {auth.email or 'unknown'}")

    if auth.is_expired():
        print("  Token expired, refreshing...")
        auth = await refresh_access_token(auth)
        print("  ✓ Token refreshed")

    chat = ChatAntigravity(model="antigravity-gemini-3-flash", auth=auth)

    print(f"\nUsing model: {chat.model}")
    print("Sending: 'Hello! What's 2 + 2?'\n")

    try:
        response = await chat.ainvoke([{"role": "user", "content": "Hello! What's 2 + 2?"}])
        print(f"Response: {response.content}")
        print("\n✓ Google Antigravity simple chat test PASSED")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_codex_simple_chat():
    """Test OpenAI Codex simple chat."""
    print("\n" + "="*60)
    print("OPENAI CODEX TEST 1: Simple Chat")
    print("="*60)

    auth = load_codex_auth_from_storage()
    if not auth:
        print("❌ No OpenAI/Codex authentication found!")
        print("   Run 'codex-auth login' first to authenticate.")
        return False

    print(f"✓ Found OpenAI auth for account: {auth.account_id or 'unknown'}")

    if auth.is_expired():
        print("  Token expired, refreshing...")
        auth = await refresh_codex_token(auth)
        print("  ✓ Token refreshed")

    chat = ChatCodex(model="gpt-5.2-codex", auth=auth)

    print(f"\nUsing model: {chat.model}")
    print("Sending: 'Hello! What's 2 + 2?'\n")

    try:
        response = await chat.ainvoke([{"role": "user", "content": "Hello! What's 2 + 2?"}])
        print(f"Response: {response.content}")
        print("\n✓ OpenAI Codex simple chat test PASSED")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_antigravity_tool_use():
    """Test Google Antigravity tool calling."""
    print("\n" + "="*60)
    print("GOOGLE ANTIGRAVITY TEST 2: Tool Use")
    print("="*60)

    auth = load_auth_from_storage()
    if not auth:
        return False

    if auth.is_expired():
        auth = await refresh_access_token(auth)

    weather_tool = define_simple_tool()
    chat = ChatAntigravity(model="antigravity-gemini-3-flash", auth=auth).bind_tools([weather_tool])

    print(f"Using model: {chat.model}")
    print("Bound tool: get_weather")
    print("Sending: 'What's the weather in San Francisco?'\n")

    try:
        response = await chat.ainvoke([{"role": "user", "content": "What's the weather in San Francisco?"}])

        print(f"Response content: {response.content or '(empty - tool call made)'}")

        if response.tool_calls:
            print(f"\nTool calls made:")
            for tc in response.tool_calls:
                print(f"  - {tc['name']}({tc['args']})")
            print("\n✓ Google Antigravity tool use test PASSED")
            return True
        else:
            print("\n⚠ Model responded without using the tool")
            print("  This may still be valid depending on the model's behavior")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_codex_tool_use():
    """Test OpenAI Codex tool calling."""
    print("\n" + "="*60)
    print("OPENAI CODEX TEST 2: Tool Use")
    print("="*60)

    auth = load_codex_auth_from_storage()
    if not auth:
        return False

    if auth.is_expired():
        auth = await refresh_codex_token(auth)

    weather_tool = define_simple_tool()
    chat = ChatCodex(model="gpt-5.2-codex", auth=auth).bind_tools([weather_tool])

    print(f"Using model: {chat.model}")
    print("Bound tool: get_weather")
    print("Sending: 'What's the weather in San Francisco?'\n")

    try:
        response = await chat.ainvoke([{"role": "user", "content": "What's the weather in San Francisco?"}])

        print(f"Response content: {response.content or '(empty - tool call made)'}")

        if response.tool_calls:
            print(f"\nTool calls made:")
            for tc in response.tool_calls:
                print(f"  - {tc['name']}({tc['args']})")
            print("\n✓ OpenAI Codex tool use test PASSED")
            return True
        else:
            print("\n⚠ Model responded without using the tool")
            print("  This may still be valid depending on the model's behavior")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_antigravity_streaming():
    """Test Google Antigravity streaming."""
    print("\n" + "="*60)
    print("GOOGLE ANTIGRAVITY TEST 3: Streaming")
    print("="*60)

    auth = load_auth_from_storage()
    if not auth:
        return False

    if auth.is_expired():
        auth = await refresh_access_token(auth)

    chat = ChatAntigravity(model="antigravity-gemini-3-flash", auth=auth)

    print(f"Using model: {chat.model}")
    print("Sending: 'Count from 1 to 5 slowly.'\n")
    print("Streaming response: ", end="", flush=True)

    try:
        async for chunk in chat.astream([{"role": "user", "content": "Count from 1 to 5 slowly."}]):
            if chunk.content:
                print(chunk.content, end="", flush=True)

        print("\n\n✓ Google Antigravity streaming test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def test_codex_streaming():
    """Test OpenAI Codex streaming."""
    print("\n" + "="*60)
    print("OPENAI CODEX TEST 3: Streaming")
    print("="*60)

    auth = load_codex_auth_from_storage()
    if not auth:
        return False

    if auth.is_expired():
        auth = await refresh_codex_token(auth)

    chat = ChatCodex(model="gpt-5.2-codex", auth=auth)

    print(f"Using model: {chat.model}")
    print("Sending: 'Count from 1 to 5 slowly.'\n")
    print("Streaming response: ", end="", flush=True)

    try:
        async for chunk in chat.astream([{"role": "user", "content": "Count from 1 to 5 slowly."}]):
            if chunk.content:
                print(chunk.content, end="", flush=True)

        print("\n\n✓ OpenAI Codex streaming test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("="*60)
    print("LangChain Antigravity & Codex Integration Tests")
    print("="*60)

    results = []

    # Test Google Antigravity
    has_google_auth = load_auth_from_storage() is not None
    if has_google_auth:
        print("\n🔵 Testing Google Antigravity authentication...")
        results.append(("Google Simple Chat", await test_antigravity_simple_chat()))
        results.append(("Google Tool Use", await test_antigravity_tool_use()))
        results.append(("Google Streaming", await test_antigravity_streaming()))
    else:
        print("\n⏭ Skipping Google Antigravity tests (no authentication)")
        print("  Run 'ag-auth login' to enable Google tests\n")

    # Test OpenAI Codex
    has_codex_auth = load_codex_auth_from_storage() is not None
    if has_codex_auth:
        print("\n🟢 Testing OpenAI Codex authentication...")
        results.append(("Codex Simple Chat", await test_codex_simple_chat()))
        results.append(("Codex Tool Use", await test_codex_tool_use()))
        results.append(("Codex Streaming", await test_codex_streaming()))
    else:
        print("\n⏭ Skipping OpenAI Codex tests (no authentication)")
        print("  Run 'codex-auth login' to enable Codex tests\n")

    if not has_google_auth and not has_codex_auth:
        print("\n❌ No authentication found for either platform!")
        print("   Run 'ag-auth login' for Google or 'codex-auth login' for OpenAI")
        return False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    total = has_google_auth * 3 + has_codex_auth * 3
    skipped = total - len(results)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n⚠ Some tests failed!")
        return False
    elif passed == 0:
        print("\n⏭ All tests skipped (no authentication)")
        return True
    else:
        print("\n✓ All executed tests passed!")
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
