import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google import genai

from app.core.config import settings


async def test_candidates():
    client = genai.Client(api_key=settings.gemini_api_key)

    candidates = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    working = []
    failed = []

    print("Testing Candidate Text Models against Live Gemini API:")
    for model_name in candidates:
        print(f"Testing model: '{model_name}'...", end=" ", flush=True)
        try:
            res = await client.aio.models.generate_content(
                model=model_name,
                contents="Reply with 'OK'",
            )
            if res.text:
                print(f"SUCCESS -> '{res.text.strip()}'")
                working.append(model_name)
            else:
                print("FAILED (Empty text)")
                failed.append((model_name, "Empty text"))
        except Exception as e:
            print(f"FAILED -> {e}")
            failed.append((model_name, str(e)))

    print("\n================ SUMMARY ================")
    print(f"Working models ({len(working)}): {working}")
    print(f"Failed models ({len(failed)}): {[f[0] for f in failed]}")


if __name__ == "__main__":
    asyncio.run(test_candidates())
