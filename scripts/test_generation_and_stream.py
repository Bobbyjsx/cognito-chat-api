import asyncio
import json
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google import genai
from google.genai import types

from app.core.config import settings
from app.utils.prompts import get_base_system_instructions


async def test_generation():
    print("==================================================")
    print("1. Testing Standard Generation (generate_content)...")
    print("==================================================")

    client = genai.Client(api_key=settings.gemini_api_key)
    model_name = "gemini-3.6-flash"

    config = types.GenerateContentConfig(
        system_instruction=get_base_system_instructions(),
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
    )

    prompt = "Explain quantum computing in 2 brief sentences."
    print(f"Prompt: '{prompt}'")
    print(f"Model: {model_name}")

    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        print("\nResponse Received:")
        print(f"Text: {response.text.strip()}")
        if response.usage_metadata:
            print(f"Total Token Count: {response.usage_metadata.total_token_count}")
        print("Status: SUCCESS ✓")
    except Exception as e:
        print(f"Status: FAILED ✗ ({e})")
        raise e


async def test_stream():
    print("\n==================================================")
    print("2. Testing Real-Time SSE Stream (generate_content_stream)...")
    print("==================================================")

    client = genai.Client(api_key=settings.gemini_api_key)
    model_name = "gemini-3.6-flash"

    config = types.GenerateContentConfig(
        system_instruction=get_base_system_instructions(),
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
    )

    prompt = "Count from 1 to 5 slowly, separated by commas."
    print(f"Prompt: '{prompt}'")
    print(f"Model: {model_name}\n")

    chunks_received = 0
    full_text = ""
    total_tokens = 0

    try:
        response_stream = await client.aio.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=config,
        )

        print("Streaming Chunks:")
        async for chunk in response_stream:
            if chunk.text:
                chunks_received += 1
                full_text += chunk.text
                event_data = json.dumps({"type": "text", "token": chunk.text})
                print(f"  [Chunk {chunks_received}] -> {event_data}")
            if chunk.usage_metadata:
                total_tokens = chunk.usage_metadata.total_token_count

        print("\nStream Summary:")
        print(f"Full Text: {full_text.strip()}")
        print(f"Total Chunks Yielded: {chunks_received}")
        print(f"Total Token Count: {total_tokens}")
        print("Status: SUCCESS ✓")
    except Exception as e:
        print(f"Status: FAILED ✗ ({e})")
        raise e


async def main():
    await test_generation()
    await test_stream()


if __name__ == "__main__":
    asyncio.run(main())
