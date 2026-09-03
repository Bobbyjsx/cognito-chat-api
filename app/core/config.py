import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()  # This explicitly loads the .env file into os.environ


class Settings(BaseSettings):
    app_name: str = "Cognito Chat API"
    debug: bool = False

    # Explicitly define this so Pydantic expects it from the .env file
    gemini_api_key: str = ""
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Claude / Anthropic Backend ("vertex" for Google Cloud Vertex AI, "anthropic" for direct API, "auto")
    claude_backend: str = "vertex"
    anthropic_vertex_project_id: str = ""
    anthropic_vertex_region: str = "us-east5"
    anthropic_vertex_credentials_path: str = ""

    # Database Configuration (Firebase)
    firebase_credentials_path: str = ""
    firestore_database: str = ""

    # Object Storage (attachments)
    # backend: "" → auto (GCS when STORAGE_BUCKET is set, otherwise local disk)
    storage_backend: str = ""
    storage_bucket: str = ""
    local_storage_dir: str = "./storage_data"

    # Auth Settings
    secret_key: str = "supersecretkey_please_change_in_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Identity Service / OAuth Resource Server Settings
    identity_service_url: str = "http://localhost:8002"
    identity_issuer: str = "http://localhost:8002"
    identity_jwks_url: str = ""
    identity_audience: str = "application_api"

    # Cache (Redis)
    redis_url: str = ""

    # Background Generation Worker Provider ("cloudtasks" or "local")
    worker_provider: str = "local"

    # Cloud Tasks
    cloud_tasks_project: str = ""
    cloud_tasks_location: str = "us-central1"
    cloud_tasks_queue: str = "cognito-generations"
    cloud_tasks_worker_url: str = ""
    cloud_tasks_service_account_email: str = ""

    # Generation Timeout (seconds)
    generation_timeout_seconds: int = 300

    environment: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
