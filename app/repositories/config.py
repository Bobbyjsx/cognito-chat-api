from fastapi import HTTPException
from google.cloud.firestore_v1.async_client import AsyncClient
from app.models.config import AppConfigDB


class ConfigRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("configs")

    async def get_config(self) -> AppConfigDB:
        """Fetches the global application configuration from Firestore.

        Raises HTTPException 500 if the app_config document has not been created by migrations.
        """
        doc_ref = self.collection.document("app_config")
        doc = await doc_ref.get()

        if not doc.exists:
            raise HTTPException(
                status_code=500,
                detail="System configuration 'configs/app_config' not found in database. Please run migrations.",
            )

        data = doc.to_dict() or {}
        return AppConfigDB(**data)

    async def update_config(self, config: AppConfigDB) -> AppConfigDB:
        doc_ref = self.collection.document("app_config")
        data = config.model_dump(mode="json")
        await doc_ref.set(data)
        return config
