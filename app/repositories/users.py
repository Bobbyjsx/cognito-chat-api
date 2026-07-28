from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient

from app.models.users import UserDB


class UserRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("users")

    async def get_by_email(self, email: str) -> UserDB | None:
        # Firestore queries return an AsyncGenerator
        docs = self.collection.where("email", "==", email).limit(1).stream()
        async for doc in docs:
            data = doc.to_dict()
            return UserDB(**data)
        return None

    async def get_by_id(self, user_id: UUID) -> UserDB | None:
        doc_ref = self.collection.document(str(user_id))
        doc = await doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return UserDB(**data)
        return None

    async def create(self, user_db: UserDB) -> UserDB:
        doc_ref = self.collection.document(str(user_db.id))
        # model_dump serializes the UUIDs and datetimes correctly if configured,
        # but Firestore handles datetimes natively. We can just use dict() or model_dump().
        # We need to make sure uuid is converted to string for Firestore.
        data = user_db.model_dump(mode="json")
        await doc_ref.set(data)
        return user_db

    async def update_token_usage(self, user_id: UUID, tokens_added: int) -> None:
        user = await self.get_by_id(user_id)
        if user:
            new_tokens = user.tokens_used + tokens_added
            doc_ref = self.collection.document(str(user_id))
            await doc_ref.update({"tokens_used": new_tokens})

    async def update_password(self, user_id: UUID, hashed_password: str) -> None:
        doc_ref = self.collection.document(str(user_id))
        await doc_ref.update({"hashed_password": hashed_password})
