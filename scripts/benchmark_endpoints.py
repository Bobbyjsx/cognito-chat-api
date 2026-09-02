import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure app is importable
sys.path.insert(0, os.path.abspath("."))

# Ensure local storage for attachments
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("LOCAL_STORAGE_DIR", "/tmp/cognito-benchmark-storage")

from fastapi.testclient import TestClient


class EndpointBenchmark:
    def __init__(self, iterations: int = 25, warmup: int = 5):
        self.iterations = iterations
        self.warmup = warmup
        self.results: Dict[str, Dict[str, Any]] = {}

    def measure(self, name: str, fn, *args, **kwargs) -> List[float]:
        # Warmup
        for _ in range(self.warmup):
            try:
                fn(*args, **kwargs)
            except Exception:
                pass

        latencies_ms: List[float] = []
        status_codes: List[int] = []

        for _ in range(self.iterations):
            start = time.perf_counter()
            resp = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            if hasattr(resp, "status_code"):
                status_codes.append(resp.status_code)

        mean_ms = statistics.mean(latencies_ms)
        median_ms = statistics.median(latencies_ms)
        stdev_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        p99_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
        min_ms = min(latencies_ms)
        max_ms = max(latencies_ms)

        self.results[name] = {
            "iterations": self.iterations,
            "mean_ms": round(mean_ms, 2),
            "median_ms": round(median_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "std_dev_ms": round(stdev_ms, 2),
            "sample_status": status_codes[0] if status_codes else None,
        }
        return latencies_ms

    def print_table(self, title: str = "API Endpoint Latency Benchmark"):
        print("\n=========================================================================================")
        print(f" {title}")
        print(f" Iterations: {self.iterations} | Warmup: {self.warmup}")
        print("=========================================================================================")
        header = f"{'Endpoint / Action':<35} | {'Mean':>8} | {'Median':>8} | {'P95':>8} | {'Min':>8} | {'Max':>8} | {'Status':>6}"
        print(header)
        print("-" * len(header))
        for name, data in self.results.items():
            print(
                f"{name:<35} | {data['mean_ms']:>6.2f}ms | {data['median_ms']:>6.2f}ms | {data['p95_ms']:>6.2f}ms | {data['min_ms']:>6.2f}ms | {data['max_ms']:>6.2f}ms | {str(data['sample_status']):>6}"
            )
        print("=========================================================================================\n")


def run_benchmarks(iterations: int = 25) -> Dict[str, Any]:
    class MockChunk:
        def __init__(self, text: str):
            self.text = text
            self.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=20, total_token_count=30)
            self.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text=text)]))]

    async def mock_generate_content(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.text = "Benchmark test response from mocked agent."
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=20, total_token_count=30)
        return mock_response

    async def mock_generate_content_stream(*args, **kwargs):
        async def _stream():
            for t in ["Benchmark ", "stream ", "response ", "tokens."]:
                yield MockChunk(t)

        return _stream()

    patch("app.core.security.verify_password", return_value=True).start()
    patch("app.core.security.get_password_hash", return_value="hashed_password").start()

    from app.main import app

    bench = EndpointBenchmark(iterations=iterations, warmup=5)

    with TestClient(app) as client:
        if hasattr(app.state, "provider") and hasattr(app.state.provider, "client"):
            app.state.provider.client.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
            app.state.provider.client.aio.models.generate_content_stream = AsyncMock(
                side_effect=mock_generate_content_stream
            )

        email = f"bench_{int(time.time())}@example.com"
        password = "secure_password_123"

        # 1. Health check
        bench.measure("GET /health", lambda: client.get("/health"))

        # 2. Config endpoint
        bench.measure("GET /config", lambda: client.get("/config"))

        # 3. Auth Signup
        bench.measure(
            "POST /auth/signup",
            lambda: client.post(
                "/auth/signup",
                json={
                    "email": f"bench_{time.time_ns()}@example.com",
                    "password": password,
                },
            ),
        )

        # 4. Auth Login
        login_payload = {"email": email, "password": password}
        client.post("/auth/signup", json=login_payload)
        login_resp = client.post("/auth/login", json=login_payload)
        auth_data = login_resp.json() if login_resp.status_code == 200 else {}
        token = auth_data.get("access_token", "")
        refresh_token = auth_data.get("refresh_token", "")
        auth_headers = {"Authorization": f"Bearer {token}"} if token else {}

        bench.measure("POST /auth/login", lambda: client.post("/auth/login", json=login_payload))

        # 5. Auth Me
        if auth_headers:
            bench.measure("GET /auth/me", lambda: client.get("/auth/me", headers=auth_headers))

        # 6. Auth Refresh
        if refresh_token:
            bench.measure(
                "POST /auth/refresh", lambda: client.post("/auth/refresh", json={"refresh_token": refresh_token})
            )

        # 7. Agent Chat (Sync)
        session_id = None
        if auth_headers:
            chat_payload = {"message": "Benchmark prompt test"}

            def _chat():
                nonlocal session_id
                r = client.post("/agent/chat", headers=auth_headers, json=chat_payload)
                if r.status_code == 200:
                    session_id = r.json().get("session_id")
                return r

            bench.measure("POST /agent/chat (Sync)", _chat)

        # 8. Agent Chat Stream (SSE)
        if auth_headers:
            stream_headers = {**auth_headers, "Accept": "text/event-stream"}
            stream_payload = {"message": "Benchmark streaming prompt"}
            bench.measure(
                "POST /agent/chat/stream",
                lambda: client.post("/agent/chat/stream", headers=stream_headers, json=stream_payload),
            )

        # 9. List Sessions
        if auth_headers:
            bench.measure("GET /agent/sessions", lambda: client.get("/agent/sessions", headers=auth_headers))

        # 10. Get Session Details
        if auth_headers and session_id:
            bench.measure(
                "GET /agent/sessions/{id}", lambda: client.get(f"/agent/sessions/{session_id}", headers=auth_headers)
            )

        # 11. Generations status (if route exists)
        if auth_headers:
            import uuid

            test_gen_uuid = str(uuid.uuid4())
            try:
                gen_resp = client.get(f"/agent/generations/{test_gen_uuid}", headers=auth_headers)
                if gen_resp.status_code in (200, 404):
                    bench.measure(
                        "GET /agent/generations/{id}",
                        lambda: client.get(f"/agent/generations/{test_gen_uuid}", headers=auth_headers),
                    )
            except Exception:
                pass

        # 12. Attachments Upload
        attachment_id = None
        if auth_headers:

            def _upload():
                nonlocal attachment_id
                files = {"file": ("bench.txt", b"Benchmark attachment content buffer", "text/plain")}
                r = client.post("/agent/attachments", headers=auth_headers, files=files)
                if r.status_code in (200, 201):
                    attachment_id = r.json().get("id")
                return r

            bench.measure("POST /agent/attachments", _upload)

        # 13. Get Attachment Metadata
        if auth_headers and attachment_id:
            bench.measure(
                "GET /agent/attachments/{id}",
                lambda: client.get(f"/agent/attachments/{attachment_id}", headers=auth_headers),
            )

        # 14. Get Attachment Content
        if auth_headers and attachment_id:
            bench.measure(
                "GET /agent/attachments/{id}/content",
                lambda: client.get(f"/agent/attachments/{attachment_id}/content", headers=auth_headers),
            )

        # 15. Delete Session
        if auth_headers and session_id:
            bench.measure(
                "DELETE /agent/sessions/{id}",
                lambda: client.delete(f"/agent/sessions/{session_id}", headers=auth_headers),
            )

    return bench.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Cognito Chat API endpoints")
    parser.add_argument("--iterations", type=int, default=25, help="Number of benchmark iterations per endpoint")
    parser.add_argument("--output", type=str, default="", help="Path to save JSON benchmark results")
    parser.add_argument("--tag", type=str, default="Current Branch", help="Tag / Branch name for display")
    args = parser.parse_args()

    results = run_benchmarks(iterations=args.iterations)

    benchmark_obj = EndpointBenchmark(iterations=args.iterations)
    benchmark_obj.results = results
    benchmark_obj.print_table(f"API Benchmark: {args.tag}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"branch": args.tag, "results": results}, f, indent=2)
        print(f"Saved benchmark results to {args.output}")
