from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.models.prompts import PromptDB


class PromptRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = db.collection("prompts")

    def _prompt_from_doc(self, data: dict[str, Any] | None) -> PromptDB | None:
        if not data:
            return None
        return PromptDB(**data)

    async def get_built_in_prompts(self) -> list[PromptDB]:
        query = self.collection.where(filter=FieldFilter("user_id", "==", None))
        prompts = []
        async for doc in query.stream():
            prompt = self._prompt_from_doc(doc.to_dict())
            if prompt:
                prompts.append(prompt)
        return prompts

    async def get_user_prompts(self, user_id: UUID | str) -> list[PromptDB]:
        query = self.collection.where(filter=FieldFilter("user_id", "==", str(user_id)))
        prompts = []
        async for doc in query.stream():
            prompt = self._prompt_from_doc(doc.to_dict())
            if prompt:
                prompts.append(prompt)
        return prompts

    async def create_prompt(self, prompt: PromptDB) -> PromptDB:
        doc_ref = self.collection.document(str(prompt.id))
        await doc_ref.set(prompt.model_dump(mode="json"))
        return prompt

    async def get_prompt(self, prompt_id: UUID | str) -> PromptDB | None:
        doc_ref = self.collection.document(str(prompt_id))
        doc = await doc_ref.get()
        if not doc.exists:
            return None
        return self._prompt_from_doc(doc.to_dict())

    async def update_prompt(self, prompt_id: UUID | str, updates: dict[str, Any]) -> None:
        doc_ref = self.collection.document(str(prompt_id))
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        await doc_ref.update(updates)

    async def delete_prompt(self, prompt_id: UUID | str) -> None:
        doc_ref = self.collection.document(str(prompt_id))
        await doc_ref.delete()
