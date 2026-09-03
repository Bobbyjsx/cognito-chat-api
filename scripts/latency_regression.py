#!/usr/bin/env python3
"""Live End-to-End Latency Regression Suite for Cognito Chat API.

Exercises all business, chat, attachment, and session endpoints directly against
a live server without mocking. If the server is not already running on port 8000,
it automatically starts an ephemeral uvicorn instance and cleanly tears it down.

All authentication endpoints are omitted. Authenticated requests use a locally
signed JWT matching the server's configured secret key.
"""

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import jwt

# Ensure app is importable
sys.path.insert(0, os.path.abspath("."))
from app.core.config import settings


def get_current_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "current-branch"


def create_benchmark_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "email": f"bench_{user_id[:8]}@example.com",
        "exp": int(time.time()) + 86400,
        "type": "access",
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


import socket


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_server_running(url: str = "http://127.0.0.1:8000") -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


class ServerManager:
    """Manages an ephemeral uvicorn server process if one isn't already running."""

    def __init__(self, port: int = 8000):
        self.default_port = port
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.process: subprocess.Popen | None = None
        self.was_running = False

    def start(self):
        if is_server_running(self.url):
            print(f"📡 Found live Cognito Chat API running on {self.url}")
            self.was_running = True
            return

        # Pick a dedicated free port for the benchmark instance
        self.port = find_free_port()
        self.url = f"http://127.0.0.1:{self.port}"

        print(f"🚀 Starting live server on {self.url} for benchmarking...")
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server to become healthy
        start_time = time.time()
        while time.time() - start_time < 25.0:
            if is_server_running(self.url):
                print(f"✅ Live server started and healthy on {self.url}")
                return
            time.sleep(0.5)

        self.stop()
        raise RuntimeError(f"Failed to start uvicorn server on {self.url} within 25 seconds.")

    def stop(self):
        if self.process and not self.was_running:
            print("🛑 Shutting down ephemeral benchmark server...")
            with contextlib.suppress(Exception):
                self.process.terminate()
                self.process.wait(timeout=5)
            with contextlib.suppress(Exception):
                if self.process and self.process.poll() is None:
                    self.process.kill()
            self.process = None


class BenchmarkCollector:
    def __init__(self, iterations: int = 15, warmup: int = 2):
        self.iterations = iterations
        self.warmup = warmup
        self.results: dict[str, dict[str, Any]] = {}

    def measure(self, name: str, fn) -> list[float]:
        for _ in range(self.warmup):
            with contextlib.suppress(Exception):
                fn()

        latencies_ms: list[float] = []
        status_codes: list[int] = []

        for _ in range(self.iterations):
            start = time.perf_counter()
            resp = fn()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            if hasattr(resp, "status_code"):
                status_codes.append(resp.status_code)

        mean_ms = statistics.mean(latencies_ms)
        median_ms = statistics.median(latencies_ms)
        stdev_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        min_ms = min(latencies_ms)
        max_ms = max(latencies_ms)

        self.results[name] = {
            "iterations": self.iterations,
            "mean_ms": round(mean_ms, 2),
            "median_ms": round(median_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "std_dev_ms": round(stdev_ms, 2),
            "sample_status": status_codes[0] if status_codes else None,
        }
        return latencies_ms


def run_live_benchmarks(base_url: str, iterations: int = 15) -> dict[str, Any]:
    collector = BenchmarkCollector(iterations=iterations, warmup=2)
    user_id = str(uuid.uuid4())
    token = create_benchmark_token(user_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        # 1. Health check
        collector.measure("GET /health", lambda: client.get("/health"))

        # 2. Config endpoint
        collector.measure("GET /config", lambda: client.get("/config", headers=headers))

        # 3. List Sessions
        collector.measure("GET /agent/sessions", lambda: client.get("/agent/sessions", headers=headers))

        # 4. Agent Chat (Sync)
        session_id = None

        def _chat():
            nonlocal session_id
            r = client.post(
                "/agent/chat",
                headers=headers,
                json={"message": "Benchmark live prompt ping"},
            )
            if r.status_code == 200:
                session_id = r.json().get("session_id")
            return r

        collector.measure("POST /agent/chat (Sync)", _chat)

        # 5. Agent Chat Stream (SSE)
        def _stream():
            with client.stream(
                "POST",
                "/agent/chat/stream",
                headers={**headers, "Accept": "text/event-stream"},
                json={"message": "Benchmark live stream prompt ping"},
            ) as r:
                for _ in r.iter_raw():
                    pass
                return r

        collector.measure("POST /agent/chat/stream", _stream)

        # 6. Get Session Details
        if session_id:
            collector.measure(
                "GET /agent/sessions/{id}",
                lambda: client.get(f"/agent/sessions/{session_id}", headers=headers),
            )

        # 7. Generations status (if endpoint exists on branch)
        test_gen_uuid = str(uuid.uuid4())
        with contextlib.suppress(Exception):
            gen_resp = client.get(f"/agent/generations/{test_gen_uuid}", headers=headers)
            if gen_resp.status_code in (200, 404):
                collector.measure(
                    "GET /agent/generations/{id}",
                    lambda: client.get(f"/agent/generations/{test_gen_uuid}", headers=headers),
                )

        # 8. Attachments Upload
        attachment_id = None

        def _upload():
            nonlocal attachment_id
            files = {"file": ("bench.txt", b"Live benchmark attachment payload bytes", "text/plain")}
            r = client.post("/agent/attachments", headers=headers, files=files)
            if r.status_code in (200, 201):
                attachment_id = r.json().get("id")
            return r

        collector.measure("POST /agent/attachments", _upload)

        # 9. Get Attachment Metadata
        if attachment_id:
            collector.measure(
                "GET /agent/attachments/{id}",
                lambda: client.get(f"/agent/attachments/{attachment_id}", headers=headers),
            )

        # 10. Get Attachment Content
        if attachment_id:
            collector.measure(
                "GET /agent/attachments/{id}/content",
                lambda: client.get(f"/agent/attachments/{attachment_id}/content", headers=headers),
            )

        # 11. Delete Session
        if session_id:
            collector.measure(
                "DELETE /agent/sessions/{id}",
                lambda: client.delete(f"/agent/sessions/{session_id}", headers=headers),
            )

    return collector.results


def generate_regression_report(
    base_file: str,
    current_results: dict[str, Any],
    current_branch: str,
    iterations: int,
) -> str:
    base_results = {}
    base_branch = "master (base)"
    if os.path.exists(base_file):
        try:
            with open(base_file, "r") as f:
                bdata = json.load(f)
                base_branch = bdata.get("branch", "master (base)")
                base_results = bdata.get("results", {})
        except Exception as e:
            print(f"Warning: Could not read base benchmark file: {e}")

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Live API Latency Regression Report",
        "",
        f"- **Generated At**: `{now_iso}`",
        f"- **Current Branch**: `{current_branch}`",
        f"- **Baseline Branch**: `{base_branch}`",
        f"- **Iterations Per Endpoint**: `{iterations}`",
        "- **Mode**: **Live E2E (Non-mocked, real DB & Gemini calls, Auth tests excluded)**",
        "",
        "---",
        "",
        "## 📊 Benchmark Latency Comparison",
        "",
        "| Endpoint / Action | Base Mean | Current Mean | Median | P95 | Delta (ms) | Delta (%) | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_keys = list(dict.fromkeys(list(base_results.keys()) + list(current_results.keys())))

    for key in all_keys:
        b_data = base_results.get(key)
        c_data = current_results.get(key)

        if not c_data:
            continue

        c_mean = c_data["mean_ms"]
        c_median = c_data["median_ms"]
        c_p95 = c_data["p95_ms"]
        http_status = c_data["sample_status"]

        if not b_data:
            lines.append(
                f"| **`{key}`** | `N/A` | `{c_mean:.2f}ms` | `{c_median:.2f}ms` | `{c_p95:.2f}ms` | `NEW` | `-` | `{http_status}` |"
            )
            continue

        b_mean = b_data["mean_ms"]
        delta_ms = c_mean - b_mean
        delta_pct = ((c_mean - b_mean) / b_mean * 100.0) if b_mean > 0 else 0.0

        sign = "+" if delta_ms > 0 else ""
        delta_str = f"{sign}{delta_ms:.2f}ms"
        pct_str = f"{sign}{delta_pct:.1f}%"

        if delta_pct > 25.0 and delta_ms > 100.0:
            badge = "⚠️ Regression"
        elif delta_pct < -5.0:
            badge = "🚀 Faster"
        else:
            badge = "✅ Nominal"

        lines.append(
            f"| **`{key}`** | `{b_mean:.2f}ms` | `{c_mean:.2f}ms` | `{c_median:.2f}ms` | `{c_p95:.2f}ms` | `{delta_str}` | `{pct_str}` | {badge} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 🔬 Observations",
            "",
            "- All measurements reflect end-to-end network requests against live Firestore & Redis.",
            "- Auth endpoints are isolated from this regression test as authentication is delegated to external Identity Service.",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run Cognito API Live Latency Regression Test")
    parser.add_argument("--iterations", type=int, default=15, help="Iterations per endpoint (default: 15)")
    parser.add_argument(
        "--base", type=str, default="benchmarks/base_benchmark.json", help="Path to base benchmark JSON"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to test / start uvicorn on")
    args = parser.parse_args()

    branch = get_current_branch()
    server = ServerManager(port=args.port)
    server.start()

    try:
        print(f"\n🚀 Running live latency regression benchmarks on branch [{branch}]...")
        print(f"⚙️  Iterations: {args.iterations} | Live Non-Mocked Execution | Auth tests excluded\n")

        current_results = run_live_benchmarks(server.url, iterations=args.iterations)

        os.makedirs("benchmarks", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        # 1. Save current benchmark JSON (ignored by git)
        current_json_path = "benchmarks/current_benchmark.json"
        with open(current_json_path, "w") as f:
            json.dump({"branch": branch, "results": current_results}, f, indent=2)
        print(f"💾 Saved current branch benchmark to: {current_json_path}")

        # 2. Generate and save Markdown regression report (ignored by git)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_md = generate_regression_report(args.base, current_results, branch, args.iterations)

        timestamped_report_path = f"reports/regression_report_{timestamp_str}.md"
        latest_report_path = "reports/latest_regression_report.md"

        with open(timestamped_report_path, "w") as f:
            f.write(report_md)
        with open(latest_report_path, "w") as f:
            f.write(report_md)

        print(f"📄 Saved regression report to: {timestamped_report_path}")
        print(f"📄 Updated latest report at:   {latest_report_path}\n")

        # 3. Print report to console
        print(report_md)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
