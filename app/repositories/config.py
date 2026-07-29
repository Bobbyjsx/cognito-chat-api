from google.cloud.firestore_v1.async_client import AsyncClient

from app.models.config import AppConfigDB


class ConfigRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("configs")

    async def get_config(self) -> AppConfigDB:
        """Fetches the global application configuration from Firestore.

        If no configuration document exists yet, initializes and persists default AppConfigDB.
        """
        doc_ref = self.collection.document("app_config")
        doc = await doc_ref.get()

        if doc.exists:
            data = doc.to_dict() or {}
            return AppConfigDB(**data)

        # Initialize default config if document does not exist
        default_config = AppConfigDB()
        await doc_ref.set(default_config.model_dump(mode="json"))
        return default_config

    async def update_config(self, config: AppConfigDB) -> AppConfigDB:
        doc_ref = self.collection.document("app_config")
        data = config.model_dump(mode="json")
        await doc_ref.set(data)
        return config
