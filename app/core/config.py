from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()  # This explicitly loads the .env file into os.environ


class Settings(BaseSettings):
    app_name: str = "Antigravity AI API"
    debug: bool = False

    # Explicitly define this so Pydantic expects it from the .env file
    gemini_api_key: str = ""

    # Database Configuration (Firebase)
    firebase_credentials_path: str = ""

    # Auth Settings
    secret_key: str = "supersecretkey_please_change_in_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Antigravity SDK configuration can go here
    system_instructions: str = "You are a helpful AI assistant."

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
