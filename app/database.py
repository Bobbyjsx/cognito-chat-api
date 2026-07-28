import os

import firebase_admin
from firebase_admin import credentials
from google.cloud.firestore_v1.async_client import AsyncClient

from app.core.config import settings


def init_db():
    if not firebase_admin._apps:
        cred = None
        # Use provided credentials file if it exists, otherwise use application default credentials
        if settings.firebase_credentials_path and os.path.exists(settings.firebase_credentials_path):
            cred = credentials.Certificate(settings.firebase_credentials_path)
            firebase_admin.initialize_app(cred)
        else:
            # Assumes GOOGLE_APPLICATION_CREDENTIALS is set or running in GCP
            firebase_admin.initialize_app()

    print("Firebase Admin initialized successfully.")


def get_db() -> AsyncClient:
    """Dependency that returns an Async Firestore client."""
    # We create an AsyncClient natively because firebase_admin's firestore.client() is synchronous
    app = firebase_admin.get_app()
    project_id = app.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        # Try to extract project ID from credential object if not explicitly set
        cred = firebase_admin.get_app().credential
        if hasattr(cred, "project_id"):
            project_id = cred.project_id

    # Create the async client
    # Note: Application Default Credentials will automatically be used by google-cloud-firestore
    database = settings.firestore_database or "(default)"
    db_client = AsyncClient(project=project_id, database=database)
    return db_client
