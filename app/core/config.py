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

    # Cache (Redis)
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    redis_url: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
