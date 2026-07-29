from datetime import datetime, timedelta, timezone
from uuid import UUID

from google.cloud.firestore_v1 import async_transaction
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.transforms import Increment

from app.models.users import UserDB
from app.utils.datetime import ensure_utc


class UserRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("users")

    async def get_by_email(self, email: str) -> UserDB | None:
        docs = self.collection.where("email", "==", email).limit(1).stream()
        async for doc in docs:
            data = doc.to_dict()
            return UserDB(**data)
        return None

    async def get_by_id(self, user_id: UUID | str) -> UserDB | None:
        doc_ref = self.collection.document(str(user_id))
        doc = await doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return UserDB(**data)
        return None

    async def create(self, user_db: UserDB) -> UserDB:
        doc_ref = self.collection.document(str(user_db.id))
        data = user_db.model_dump(mode="json")
        await doc_ref.set(data)
        return user_db

    async def update_token_usage(self, user_id: UUID | str, tokens_added: int) -> None:
        """Atomically increments tokens_used using a server-side transform."""
        doc_ref = self.collection.document(str(user_id))
        await doc_ref.update({"tokens_used": Increment(tokens_added)})

    async def atomic_increment_if_within_limit(self, user_id: UUID | str, tokens_added: int) -> bool:
        """Atomically checks both the 6-hourly and weekly windows, resets them
        if their timestamps have passed (in UTC), then increments all counters if within limits.

        Returns True if the increment was applied, False if either quota is exceeded.
        """
        doc_ref = self.collection.document(str(user_id))

        @async_transaction.async_transactional
        async def _txn(transaction, doc_ref):
            snapshot = await doc_ref.get(transaction=transaction)
            data = snapshot.to_dict() or {}
            now = datetime.now(timezone.utc)

            tokens_used_6h = data.get("tokens_used_6h", 0)
            token_limit_6h = data.get("token_limit_6h", 60_000)
            reset_at = ensure_utc(data.get("reset_at"))

            tokens_used_weekly = data.get("tokens_used_weekly", 0)
            token_limit_weekly = data.get("token_limit_weekly", 300_000)
            weekly_reset_at = ensure_utc(data.get("weekly_reset_at"))

            updates: dict = {}

            # Reset 6h window if expired
            if reset_at is None or now >= reset_at:
                tokens_used_6h = 0
                updates["tokens_used_6h"] = 0
                updates["reset_at"] = (now + timedelta(hours=6)).isoformat()

            # Reset weekly window if expired
            if weekly_reset_at is None or now >= weekly_reset_at:
                tokens_used_weekly = 0
                updates["tokens_used_weekly"] = 0
                updates["weekly_reset_at"] = (now + timedelta(weeks=1)).isoformat()

            # Guard both windows
            if tokens_used_6h + tokens_added > token_limit_6h:
                return False
            if tokens_used_weekly + tokens_added > token_limit_weekly:
                return False

            # Atomically apply resets + increments together
            updates["tokens_used"] = Increment(tokens_added)
            updates["tokens_used_6h"] = Increment(tokens_added)
            updates["tokens_used_weekly"] = Increment(tokens_added)
            transaction.update(doc_ref, updates)
            return True

        return await _txn(self.db.transaction(), doc_ref)

    async def update_password(self, user_id: UUID | str, hashed_password: str) -> None:
        doc_ref = self.collection.document(str(user_id))
        await doc_ref.update({"hashed_password": hashed_password})
