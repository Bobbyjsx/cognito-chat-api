from datetime import datetime, timedelta, timezone
from typing import Any
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

    async def _sync_and_load_user(self, doc_snapshot: Any) -> UserDB | None:
        """Loads a user from a Firestore document snapshot.

        If legacy/missing fields exist or if reset windows have expired,
        synchronizes and persists the updated reset timestamps and counters
        to Firestore immediately so data is strictly persistent.
        """
        if not doc_snapshot or not doc_snapshot.exists:
            return None

        data = doc_snapshot.to_dict() or {}
        now = datetime.now(timezone.utc)
        need_update = False
        updates: dict[str, Any] = {}

        # 6-hourly window sync
        reset_at = ensure_utc(data.get("reset_at"))
        if reset_at is None or now >= reset_at:
            new_reset = now + timedelta(hours=6)
            data["reset_at"] = new_reset.isoformat()
            data["tokens_used_6h"] = 0
            updates["reset_at"] = new_reset.isoformat()
            updates["tokens_used_6h"] = 0
            need_update = True

        # Weekly window sync
        weekly_reset_at = ensure_utc(data.get("weekly_reset_at"))
        if weekly_reset_at is None or now >= weekly_reset_at:
            new_weekly_reset = now + timedelta(weeks=1)
            data["weekly_reset_at"] = new_weekly_reset.isoformat()
            data["tokens_used_weekly"] = 0
            updates["weekly_reset_at"] = new_weekly_reset.isoformat()
            updates["tokens_used_weekly"] = 0
            need_update = True

        if need_update:
            await doc_snapshot.reference.update(updates)

        return UserDB(**data)

    async def get_by_email(self, email: str) -> UserDB | None:
        docs = self.collection.where("email", "==", email).limit(1).stream()
        async for doc in docs:
            return await self._sync_and_load_user(doc)
        return None

    async def get_by_id(self, user_id: UUID | str) -> UserDB | None:
        doc_ref = self.collection.document(str(user_id))
        doc = await doc_ref.get()
        return await self._sync_and_load_user(doc)

    async def create(self, user_db: UserDB) -> UserDB:
        doc_ref = self.collection.document(str(user_db.id))
        data = user_db.model_dump(mode="json")
        await doc_ref.set(data)
        return user_db

    async def update_token_usage(self, user_id: UUID | str, tokens_added: int) -> None:
        """Atomically increments tokens_used using a server-side transform."""
        doc_ref = self.collection.document(str(user_id))
        await doc_ref.update({"tokens_used": Increment(tokens_added)})

    async def atomic_increment_if_within_limit(
        self,
        user_id: UUID | str,
        tokens_added: int,
        default_limit_6h: int = 60_000,
        default_limit_weekly: int = 300_000,
    ) -> bool:
        """Atomically checks both 6-hourly and weekly windows inside a Firestore transaction.

        If either window has expired, resets the window counters and advances the reset
        timestamps permanently in Firestore before verifying quotas and applying increments.

        Returns True if the increment was applied, False if either limit is exceeded.
        """
        doc_ref = self.collection.document(str(user_id))

        @async_transaction.async_transactional
        async def _txn(transaction, doc_ref):
            snapshot = await doc_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False
            data = snapshot.to_dict() or {}
            now = datetime.now(timezone.utc)

            updates: dict[str, Any] = {}

            # 6-hourly reset check
            reset_at = ensure_utc(data.get("reset_at"))
            tokens_used_6h = data.get("tokens_used_6h", 0)
            user_limit_6h = data.get("token_limit_6h")
            token_limit_6h = user_limit_6h if user_limit_6h is not None else default_limit_6h

            if reset_at is None or now >= reset_at:
                tokens_used_6h = 0
                updates["tokens_used_6h"] = 0
                updates["reset_at"] = (now + timedelta(hours=6)).isoformat()

            # Weekly reset check
            weekly_reset_at = ensure_utc(data.get("weekly_reset_at"))
            tokens_used_weekly = data.get("tokens_used_weekly", 0)
            user_limit_weekly = data.get("token_limit_weekly")
            token_limit_weekly = user_limit_weekly if user_limit_weekly is not None else default_limit_weekly

            if weekly_reset_at is None or now >= weekly_reset_at:
                tokens_used_weekly = 0
                updates["tokens_used_weekly"] = 0
                updates["weekly_reset_at"] = (now + timedelta(weeks=1)).isoformat()

            # Check quota limits
            if tokens_used_6h + tokens_added > token_limit_6h:
                return False
            if tokens_used_weekly + tokens_added > token_limit_weekly:
                return False

            # Apply atomic increments
            updates["tokens_used"] = Increment(tokens_added)
            updates["tokens_used_6h"] = Increment(tokens_added)
            updates["tokens_used_weekly"] = Increment(tokens_added)

            transaction.update(doc_ref, updates)
            return True

        return await _txn(self.db.transaction(), doc_ref)

    async def update_password(self, user_id: UUID | str, hashed_password: str) -> None:
        doc_ref = self.collection.document(str(user_id))
        await doc_ref.update({"hashed_password": hashed_password})
