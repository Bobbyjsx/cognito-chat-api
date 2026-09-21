from google.cloud import firestore
from google.cloud.firestore_v1.async_client import AsyncClient

from app.billing.models import SubscriptionDB


class SubscriptionRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = db.collection("subscriptions")

    async def get_by_user_id(self, user_id: str) -> SubscriptionDB | None:
        docs = (
            await self.collection.where(filter=firestore.FieldFilter("user_id", "==", user_id))
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(1)
            .get()
        )

        if not docs:
            return None
        data = docs[0].to_dict()
        data["id"] = docs[0].id
        return SubscriptionDB(**data)

    async def get_by_id(self, subscription_id: str) -> SubscriptionDB | None:
        doc = await self.collection.document(subscription_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        data["id"] = doc.id
        return SubscriptionDB(**data)

    async def get_by_provider_subscription_code(self, code: str) -> SubscriptionDB | None:
        docs = (
            await self.collection.where(filter=firestore.FieldFilter("provider_subscription_code", "==", code))
            .limit(1)
            .get()
        )

        if not docs:
            return None
        data = docs[0].to_dict()
        data["id"] = docs[0].id
        return SubscriptionDB(**data)

    async def get_by_provider_customer_code(self, code: str) -> SubscriptionDB | None:
        docs = (
            await self.collection.where(filter=firestore.FieldFilter("provider_customer_code", "==", code))
            .limit(1)
            .get()
        )

        if not docs:
            return None
        data = docs[0].to_dict()
        data["id"] = docs[0].id
        return SubscriptionDB(**data)

    async def save(self, subscription: SubscriptionDB) -> SubscriptionDB:
        data = subscription.model_dump(exclude={"id"})
        if not subscription.id:
            import uuid

            subscription.id = str(uuid.uuid4())

        doc_ref = self.collection.document(subscription.id)
        await doc_ref.set(data)

        try:
            from app.core.cache_keys import CacheKeys
            from app.core.redis import redis_cache

            await redis_cache.delete(CacheKeys.user_profile(subscription.user_id))
            await redis_cache.delete(CacheKeys.user_auth(subscription.user_id))
        except Exception:  # noqa: S110
            pass

        return subscription

    async def mark_event_processed(self, event_id: str) -> bool:
        doc_ref = self.db.collection("payment_events").document(event_id)
        from google.api_core.exceptions import AlreadyExists

        try:
            await doc_ref.create({"processed_at": firestore.SERVER_TIMESTAMP})
            return True
        except AlreadyExists:
            return False
