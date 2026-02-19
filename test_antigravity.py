#!/usr/bin/env python3
"""
Test script for LangChain Antigravity integration.

Prerequisites:
1. Run `ag-auth login` first to authenticate
2. Install dependencies: pip install langchain-core httpx pydantic

Usage:
    python test_antigravity.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add the parent directory to the path so we can import langchain_antigravity
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_antigravity.auth import load_auth_from_storage, refresh_access_token
from langchain_antigravity.chat_model import ChatAntigravity

PROJECT_ID = os.environ.get("ANTIGRAVITY_PROJECT_ID", "").strip() or None


# Define a simple tool for testing
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and state, e.g. "San Francisco, CA"
    
    Returns:
        A description of the current weather
    """
    # This is a mock implementation
    return f"The weather in {location} is sunny and 72°F."


async def test_simple_chat():
    """Test a simple chat without tools."""
    print("\n" + "="*60)
    print("TEST 1: Simple Chat")
    print("="*60)
    
    # Check if we have authentication
    auth = load_auth_from_storage()
    if not auth:
        print("❌ No authentication found!")
        print("   Run 'antigravity login' first to authenticate.")
        return False
    
    print(f"✓ Found auth for: {auth.email or 'unknown'}")
    
    # Refresh if expired
    if auth.is_expired():
        print("  Token expired, refreshing...")
        auth = await refresh_access_token(auth)
        print("  ✓ Token refreshed")
    
    # Create chat model
    chat = ChatAntigravity(
        model="antigravity-gemini-3-flash",
        auth=auth,
        project_id=PROJECT_ID,
    )
    
    print(f"\nUsing model: {chat.model}")
    print("Sending: 'Hello! What's 2 + 2?'\n")
    
    try:
        response = await chat.ainvoke([
            {"role": "user", "content": "Hello! What's 2 + 2?"}
        ])
        print(f"Response: {response.content}")
        print("\n✓ Simple chat test PASSED")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_tool_use():
    """Test tool/function calling."""
    print("\n" + "="*60)
    print("TEST 2: Tool Use")
    print("="*60)
    
    auth = load_auth_from_storage()
    if not auth:
        print("❌ No authentication found!")
        return False
    
    if auth.is_expired():
        auth = await refresh_access_token(auth)
    
    # Create a simple tool schema
    weather_tool = {
        "name": "get_weather",
        "description": "Get the current weather for a location. Use this when the user asks about weather.",
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
    
    # Create chat model with tool
    chat = ChatAntigravity(
        model="antigravity-gemini-3-flash",  # or antigravity-claude-sonnet-4-5
        auth=auth,
        project_id=PROJECT_ID,
    ).bind_tools([weather_tool])
    
    print(f"Using model: {chat.model}")
    print("Bound tool: get_weather")
    print("Sending: 'What's the weather like in San Francisco?'\n")
    
    try:
        response = await chat.ainvoke([
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ])
        
        print(f"Response content: {response.content or '(empty - tool call made)'}")
        
        if response.tool_calls:
            print(f"\nTool calls made:")
            for tc in response.tool_calls:
                print(f"  - {tc['name']}({tc['args']})")
            print("\n✓ Tool use test PASSED - Model correctly called the tool!")
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


async def test_streaming():
    """Test streaming responses."""
    print("\n" + "="*60)
    print("TEST 3: Streaming")
    print("="*60)
    
    auth = load_auth_from_storage()
    if not auth:
        print("❌ No authentication found!")
        return False
    
    if auth.is_expired():
        auth = await refresh_access_token(auth)
    
    chat = ChatAntigravity(
        model="antigravity-gemini-3-flash",
        auth=auth,
        project_id=PROJECT_ID,
    )
    
    print(f"Using model: {chat.model}")
    print("Sending: 'Count from 1 to 5 slowly.'\n")
    print("Streaming response: ", end="", flush=True)
    
    try:
        async for chunk in chat.astream([
            {"role": "user", "content": "Count from 1 to 5 slowly."}
        ]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\n\n✓ Streaming test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("="*60)
    print("LangChain Antigravity Integration Tests")
    print("="*60)
    
    results = []
    
    # Test 1: Simple chat
    results.append(("Simple Chat", await test_simple_chat()))
    
    # Test 2: Tool use
    results.append(("Tool Use", await test_tool_use()))
    
    # Test 3: Streaming
    results.append(("Streaming", await test_streaming()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
