"""Wake identity-service as soon as the container starts.

Stdlib only so this runs before FastAPI/Firebase imports. That overlaps the
identity cold start with this process's import time (~6s on Cloud Run).
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request


def jwks_url() -> str:
    explicit = os.environ.get("IDENTITY_JWKS_URL", "").strip()
    if explicit:
        return explicit
    base = os.environ.get("IDENTITY_SERVICE_URL", "").strip().rstrip("/")
    if not base or "localhost" in base or "127.0.0.1" in base:
        return ""
    return f"{base}/.well-known/jwks.json"


def main() -> None:
    url = jwks_url()
    if not url:
        return
    try:
        urllib.request.urlopen(url, timeout=20)
    except (urllib.error.URLError, TimeoutError, OSError):
        return


if __name__ == "__main__":
    main()
