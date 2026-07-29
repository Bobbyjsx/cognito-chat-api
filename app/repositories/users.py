from uuid import UUID

from google.cloud.firestore_v1 import async_transaction
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.transforms import Increment

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
        """Atomically increments tokens_used using a server-side transform."""
        doc_ref = self.collection.document(str(user_id))
        await doc_ref.update({"tokens_used": Increment(tokens_added)})

    async def atomic_increment_if_within_limit(self, user_id: UUID, tokens_added: int) -> bool:
        """Atomically adds tokens only if the user is still within their limit.

        Re-reads the live Firestore count inside a transaction so concurrent
        requests cannot both slip through a stale in-memory check.

        Returns True if the increment was applied, False if the limit was hit.
        """
        doc_ref = self.collection.document(str(user_id))

        @async_transaction.async_transactional
        async def _txn(transaction, doc_ref):
            snapshot = await doc_ref.get(transaction=transaction)
            data = snapshot.to_dict() or {}
            tokens_used = data.get("tokens_used", 0)
            token_limit = data.get("token_limit", 0)
            if tokens_used + tokens_added > token_limit:
                return False
            transaction.update(doc_ref, {"tokens_used": Increment(tokens_added)})
            return True

        return await _txn(self.db.transaction(), doc_ref)

    async def update_password(self, user_id: UUID, hashed_password: str) -> None:
        doc_ref = self.collection.document(str(user_id))
        await doc_ref.update({"hashed_password": hashed_password})
