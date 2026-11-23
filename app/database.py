import json
import logging
import os
from typing import Any

import firebase_admin
from fastapi import Request
from firebase_admin import credentials
from google.cloud.firestore_v1.async_client import AsyncClient

from app.core.config import settings

logger = logging.getLogger(__name__)


def load_service_account_info() -> dict[str, Any] | None:
    """Loads Google / Firebase service account info from JSON string or file."""
    # 1. Direct JSON string from environment or settings
    raw_json = (
        os.environ.get("FIREBASE_CREDENTIALS_JSON")
        or os.environ.get("GCP_SERVICE_KEY")
        or getattr(settings, "firebase_credentials_json", "")
        or getattr(settings, "gcp_service_key", "")
    )
    if raw_json and raw_json.strip():
        try:
            return json.loads(raw_json.strip())
        except Exception as exc:
            logger.warning("Failed to parse FIREBASE_CREDENTIALS_JSON: %s", exc)

    # 2. File path candidates
    candidates = [
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        getattr(settings, "firebase_credentials_path", ""),
        "firebase-credentials.json",
        "/app/firebase-credentials.json",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Failed to read credentials file '%s': %s", candidate, exc)

    return None


def init_db():
    if not firebase_admin._apps:
        # Skip credential initialization if using local emulator
        if os.environ.get("FIRESTORE_EMULATOR_HOST"):
            try:
                firebase_admin.initialize_app()
                logger.info("Firebase Admin initialized for emulator.")
                return
            except Exception as exc:
                logger.info("Firebase emulator init: %s", exc)
                return

        sa_info = load_service_account_info()
        if sa_info:
            cred = credentials.Certificate(sa_info)
            firebase_admin.initialize_app(cred)
            tmp_path = "/tmp/firebase-credentials.json"
            if not os.path.exists(tmp_path):
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(sa_info, f)
                    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", tmp_path)
                except Exception as exc:
                    logger.debug("Could not write /tmp/firebase-credentials.json: %s", exc)
        else:
            try:
                firebase_admin.initialize_app()
            except Exception as exc:
                logger.warning("Firebase default init error: %s", exc)

    logger.info("Firebase Admin initialized successfully.")


def create_db_client() -> AsyncClient:
    app = firebase_admin.get_app()
    project_id = app.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        cred = firebase_admin.get_app().credential
        if hasattr(cred, "project_id"):
            project_id = cred.project_id

    database = settings.firestore_database or "(default)"
    return AsyncClient(project=project_id, database=database)


def get_db(request: Request) -> AsyncClient:
    """Dependency that returns an Async Firestore client from app state."""
    db = getattr(request.app.state, "db_client", None)
    if db is None:
        db = create_db_client()
        request.app.state.db_client = db
    return db
