"""Tests for 6-hourly and weekly token quota enforcement.

All time-dependent tests freeze `datetime.now` via `freezegun` so we can
simulate window expiry without actually waiting.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.config import AppConfigDB
from app.models.users import UserDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(**kwargs) -> UserDB:
    now = datetime.now(timezone.utc)
    defaults = {
        "email": "test@example.com",
        "hashed_password": "hashed",
        "tokens_used": 0,
        "tokens_used_6h": 0,
        "token_limit_6h": 60_000,
        "reset_at": now + timedelta(hours=6),
        "tokens_used_weekly": 0,
        "token_limit_weekly": 300_000,
        "weekly_reset_at": now + timedelta(weeks=1),
    }
    defaults.update(kwargs)
    return UserDB(**defaults)


def _auth_headers(client, email="quota@example.com", password="password123"):
    from app.core.security import create_access_token

    user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, email))
    token = create_access_token(data={"sub": user_id, "email": email})
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Unit tests — UserDB model defaults
# ---------------------------------------------------------------------------


class TestUserDBDefaults:
    def test_6h_reset_defaults_to_6h_from_now(self):
        before = datetime.now(timezone.utc)
        user = UserDB(email="a@b.com", hashed_password="x")
        after = datetime.now(timezone.utc)
        assert before + timedelta(hours=6) <= user.reset_at <= after + timedelta(hours=6)

    def test_weekly_reset_defaults_to_7_days_from_now(self):
        before = datetime.now(timezone.utc)
        user = UserDB(email="a@b.com", hashed_password="x")
        after = datetime.now(timezone.utc)
        assert before + timedelta(weeks=1) <= user.weekly_reset_at <= after + timedelta(weeks=1)

    def test_6h_token_limit_default_is_none_for_global(self):
        user = UserDB(email="a@b.com", hashed_password="x")
        assert user.token_limit_6h is None

    def test_weekly_token_limit_default_is_none_for_global(self):
        user = UserDB(email="a@b.com", hashed_password="x")
        assert user.token_limit_weekly is None


# ---------------------------------------------------------------------------
# Unit tests — Repository transaction logic (mocked Firestore)
# ---------------------------------------------------------------------------


class TestAtomicIncrementTransaction:
    """Tests for UserRepository.atomic_increment_if_within_limit."""

    @pytest.mark.asyncio
    async def test_allows_increment_within_both_limits(self):
        import uuid

        from app.repositories.users import UserRepository

        now = datetime.now(timezone.utc)
        doc_data = {
            "tokens_used": 100,
            "tokens_used_6h": 100,
            "token_limit_6h": 60_000,
            "reset_at": (now + timedelta(hours=6)).isoformat(),
            "tokens_used_weekly": 100,
            "token_limit_weekly": 300_000,
            "weekly_reset_at": (now + timedelta(weeks=1)).isoformat(),
        }

        snapshot = MagicMock()
        snapshot.to_dict.return_value = doc_data
        doc_ref = MagicMock()
        doc_ref.get = AsyncMock(return_value=snapshot)
        collection = MagicMock()
        collection.document.return_value = doc_ref
        mock_txn = MagicMock()
        db = MagicMock()
        db.collection.return_value = collection
        db.transaction.return_value = mock_txn

        with patch("google.cloud.firestore_v1.async_transaction.async_transactional", lambda f: f):
            repo = UserRepository(db)
            result = await repo.atomic_increment_if_within_limit(uuid.uuid4(), 500)
            assert result is True

    @pytest.mark.asyncio
    async def test_rejects_when_6h_limit_exceeded(self):
        """If tokens_used_6h is already at the cap, return False."""
        import uuid

        from app.repositories.users import UserRepository

        now = datetime.now(timezone.utc)
        doc_data = {
            "tokens_used": 0,
            "tokens_used_6h": 59_999,
            "token_limit_6h": 60_000,
            "reset_at": (now + timedelta(hours=6)).isoformat(),
            "tokens_used_weekly": 0,
            "token_limit_weekly": 300_000,
            "weekly_reset_at": (now + timedelta(weeks=1)).isoformat(),
        }

        snapshot = MagicMock()
        snapshot.to_dict.return_value = doc_data
        doc_ref = MagicMock()
        doc_ref.get = AsyncMock(return_value=snapshot)
        collection = MagicMock()
        collection.document.return_value = doc_ref
        mock_txn = MagicMock()
        db = MagicMock()
        db.collection.return_value = collection
        db.transaction.return_value = mock_txn

        with patch("google.cloud.firestore_v1.async_transaction.async_transactional", lambda f: f):
            repo = UserRepository(db)
            result = await repo.atomic_increment_if_within_limit(uuid.uuid4(), 2)
            # 59_999 + 2 = 60_001 > 60_000 → False
            assert result is False

    @pytest.mark.asyncio
    async def test_resets_6h_window_when_expired(self):
        """When reset_at is in the past, tokens_used_6h should be zeroed before checking."""
        import uuid

        from app.repositories.users import UserRepository

        past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        doc_data = {
            "tokens_used": 0,
            "tokens_used_6h": 59_999,  # Would normally block, but window is expired
            "token_limit_6h": 60_000,
            "reset_at": past,
            "tokens_used_weekly": 0,
            "token_limit_weekly": 300_000,
            "weekly_reset_at": (datetime.now(timezone.utc) + timedelta(weeks=1)).isoformat(),
        }

        snapshot = MagicMock()
        snapshot.to_dict.return_value = doc_data
        doc_ref = MagicMock()
        doc_ref.get = AsyncMock(return_value=snapshot)
        collection = MagicMock()
        collection.document.return_value = doc_ref
        mock_txn = MagicMock()
        db = MagicMock()
        db.collection.return_value = collection
        db.transaction.return_value = mock_txn

        with patch("google.cloud.firestore_v1.async_transaction.async_transactional", lambda f: f):
            repo = UserRepository(db)
            result = await repo.atomic_increment_if_within_limit(uuid.uuid4(), 500)
            assert result is True

    @pytest.mark.asyncio
    async def test_resets_weekly_window_when_expired(self):
        """When weekly_reset_at is in the past, tokens_used_weekly should be zeroed."""
        import uuid

        from app.repositories.users import UserRepository

        past_weekly = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        doc_data = {
            "tokens_used": 0,
            "tokens_used_6h": 0,
            "token_limit_6h": 60_000,
            "reset_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
            "tokens_used_weekly": 299_999,  # Would normally block
            "token_limit_weekly": 300_000,
            "weekly_reset_at": past_weekly,
        }

        snapshot = MagicMock()
        snapshot.to_dict.return_value = doc_data
        doc_ref = MagicMock()
        doc_ref.get = AsyncMock(return_value=snapshot)
        collection = MagicMock()
        collection.document.return_value = doc_ref
        mock_txn = MagicMock()
        db = MagicMock()
        db.collection.return_value = collection
        db.transaction.return_value = mock_txn

        with patch("google.cloud.firestore_v1.async_transaction.async_transactional", lambda f: f):
            repo = UserRepository(db)
            result = await repo.atomic_increment_if_within_limit(uuid.uuid4(), 500)
            assert result is True


# ---------------------------------------------------------------------------
# Integration tests — HTTP endpoints via TestClient
# ---------------------------------------------------------------------------


class TestTokenQuotaEndpoints:
    def test_chat_blocked_when_6h_limit_reached(self, client, mock_agent):
        from app.models.config import AppConfigDB

        headers = _auth_headers(client, "quota6h@example.com")

        now = datetime.now(timezone.utc)
        exhausted_user = _make_user(
            email="quota6h@example.com",
            tokens_used_6h=60_000,
            token_limit_6h=60_000,
            reset_at=now + timedelta(hours=6),
        )

        with (
            patch("app.api.dependencies.UserRepository") as MockRepo,
            patch(
                "app.repositories.config.ConfigRepository.get_config",
                new=AsyncMock(return_value=AppConfigDB(default_token_limit_6h=60_000)),
            ),
            patch(
                "app.repositories.chats.ChatRepository.create_session",
                new=AsyncMock(return_value=MagicMock(id=uuid.uuid4(), messages=[])),
            ),
        ):
            MockRepo.return_value.get_by_id = AsyncMock(return_value=exhausted_user)
            resp = client.post("/agent/chat", headers=headers, json={"message": "hi"})

        assert resp.status_code == 429
        assert "6-hour" in resp.json()["detail"]

    def test_chat_blocked_when_weekly_limit_reached(self, client, mock_agent):
        from app.models.config import AppConfigDB

        headers = _auth_headers(client, "quotawk@example.com")

        now = datetime.now(timezone.utc)
        exhausted_user = _make_user(
            email="quotawk@example.com",
            tokens_used_weekly=300_000,
            token_limit_weekly=300_000,
            weekly_reset_at=now + timedelta(weeks=1),
        )

        with (
            patch("app.api.dependencies.UserRepository") as MockRepo,
            patch(
                "app.repositories.config.ConfigRepository.get_config",
                new=AsyncMock(return_value=AppConfigDB(default_token_limit_weekly=300_000)),
            ),
            patch(
                "app.repositories.chats.ChatRepository.create_session",
                new=AsyncMock(return_value=MagicMock(id=uuid.uuid4(), messages=[])),
            ),
        ):
            MockRepo.return_value.get_by_id = AsyncMock(return_value=exhausted_user)
            resp = client.post("/agent/chat", headers=headers, json={"message": "hi"})

        assert resp.status_code == 429
        assert "Weekly" in resp.json()["detail"]

    def test_chat_allowed_after_6h_window_expires(self, client, mock_agent):
        from app.models.config import AppConfigDB

        headers = _auth_headers(client, "quotaexp@example.com")

        now = datetime.now(timezone.utc)
        # Window has ALREADY expired
        user_past_reset = _make_user(
            email="quotaexp@example.com",
            tokens_used_6h=60_000,
            token_limit_6h=60_000,
            reset_at=now - timedelta(seconds=1),  # expired
        )

        with (
            patch("app.api.dependencies.UserRepository") as MockAuthRepo,
            patch("app.router.chats.UserRepository") as MockChatUserRepo,
            patch("app.repositories.config.ConfigRepository.get_config", new=AsyncMock(return_value=AppConfigDB())),
            patch(
                "app.repositories.chats.ChatRepository.create_session",
                new=AsyncMock(return_value=MagicMock(id=uuid.uuid4(), messages=[])),
            ),
            patch(
                "app.repositories.chats.ChatRepository.add_message",
                new=AsyncMock(return_value=MagicMock(id=uuid.uuid4())),
            ),
        ):
            MockAuthRepo.return_value.get_by_id = AsyncMock(return_value=user_past_reset)
            # AgentService uses a separate UserRepository instance for quota increment
            MockChatUserRepo.return_value.atomic_increment_if_within_limit = AsyncMock(return_value=True)
            resp = client.post("/agent/chat", headers=headers, json={"message": "hi"})

        # Pre-gen check should pass (window expired → effective_6h = 0)
        assert resp.status_code == 200

    def test_me_exposes_quota_fields(self, client):
        headers = _auth_headers(client, "quotame@example.com")
        mock_user = _make_user(email="quotame@example.com")
        with (
            patch("app.api.dependencies.UserRepository") as MockRepo,
            patch("app.repositories.config.ConfigRepository.get_config", new=AsyncMock(return_value=AppConfigDB())),
        ):
            MockRepo.return_value.get_by_id = AsyncMock(return_value=mock_user)
            resp = client.get("/auth/me", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "pct_6h" in data
        assert "pct_weekly" in data
        assert "reset_at" in data
        assert "weekly_reset_at" in data

    def test_reset_at_is_6h_from_signup(self, client):
        frozen_now = datetime(2026, 7, 29, 6, 0, 0, tzinfo=timezone.utc)
        mock_user = _make_user(
            email="quotareset@example.com",
            reset_at=frozen_now + timedelta(hours=6),
            weekly_reset_at=frozen_now + timedelta(weeks=1),
        )
        with (
            patch("app.api.dependencies.UserRepository") as MockRepo,
            patch("app.repositories.config.ConfigRepository.get_config", new=AsyncMock(return_value=AppConfigDB())),
        ):
            MockRepo.return_value.get_by_id = AsyncMock(return_value=mock_user)
            headers = _auth_headers(client, "quotareset@example.com")
            me = client.get("/auth/me", headers=headers).json()
        reset_at = datetime.fromisoformat(me["reset_at"])
        expected = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
        assert reset_at == expected

    def test_weekly_reset_at_is_7_days_from_signup(self, client):
        frozen_now = datetime(2026, 7, 29, 6, 0, 0, tzinfo=timezone.utc)
        mock_user = _make_user(
            email="quotawkreset@example.com",
            reset_at=frozen_now + timedelta(hours=6),
            weekly_reset_at=frozen_now + timedelta(weeks=1),
        )
        with (
            patch("app.api.dependencies.UserRepository") as MockRepo,
            patch("app.repositories.config.ConfigRepository.get_config", new=AsyncMock(return_value=AppConfigDB())),
        ):
            MockRepo.return_value.get_by_id = AsyncMock(return_value=mock_user)
            headers = _auth_headers(client, "quotawkreset@example.com")
            me = client.get("/auth/me", headers=headers).json()
        weekly_reset_at = datetime.fromisoformat(me["weekly_reset_at"])
        expected = datetime(2026, 8, 5, 6, 0, 0, tzinfo=timezone.utc)
        assert weekly_reset_at == expected

    def test_reset_timestamp_is_persistent_across_multiple_requests(self, client):
        mock_user = _make_user(email="persist@example.com")
        headers = _auth_headers(client, "persist@example.com")
        with (
            patch("app.api.dependencies.UserRepository") as MockRepo,
            patch("app.repositories.config.ConfigRepository.get_config", new=AsyncMock(return_value=AppConfigDB())),
        ):
            MockRepo.return_value.get_by_id = AsyncMock(return_value=mock_user)
            res1 = client.get("/auth/me", headers=headers).json()
            res2 = client.get("/auth/me", headers=headers).json()
            res3 = client.get("/auth/me", headers=headers).json()

        assert res1["reset_at"] == res2["reset_at"] == res3["reset_at"]
        assert res1["weekly_reset_at"] == res2["weekly_reset_at"] == res3["weekly_reset_at"]


# ---------------------------------------------------------------------------
# Edge Case Tests — Cache Invalidation on Quota Breach & Early Rejection
# ---------------------------------------------------------------------------


class TestQuotaBreachCacheInvalidation:
    """Ensures Redis cache is ALWAYS invalidated when a user breaches quota,
    preventing stale usage data from allowing infinite bypass loops."""

    @pytest.mark.asyncio
    async def test_atomic_increment_invalidates_cache_even_when_limit_exceeded(self):
        """When atomic_increment_if_within_limit exceeds limit and returns False,
        Redis cache (user_profile and user_auth) must still be deleted so stale
        quota states are never served to subsequent requests."""
        from app.repositories.users import UserRepository

        now = datetime.now(timezone.utc)
        test_uid = uuid.uuid4()
        doc_data = {
            "tokens_used": 0,
            "tokens_used_6h": 60_000,
            "token_limit_6h": 60_000,
            "reset_at": (now + timedelta(hours=6)).isoformat(),
            "tokens_used_weekly": 0,
            "token_limit_weekly": 300_000,
            "weekly_reset_at": (now + timedelta(weeks=1)).isoformat(),
        }

        snapshot = MagicMock()
        snapshot.to_dict.return_value = doc_data
        doc_ref = MagicMock()
        doc_ref.get = AsyncMock(return_value=snapshot)
        collection = MagicMock()
        collection.document.return_value = doc_ref
        mock_txn = MagicMock()
        db = MagicMock()
        db.collection.return_value = collection
        db.transaction.return_value = mock_txn

        with (
            patch("google.cloud.firestore_v1.async_transaction.async_transactional", lambda f: f),
            patch("app.core.redis.redis_cache.delete", new=AsyncMock()) as mock_delete,
        ):
            repo = UserRepository(db)
            result = await repo.atomic_increment_if_within_limit(test_uid, 100)
            assert result is False  # Exceeded limit
            # Cache invalidation MUST be called for both user_profile and user_auth
            deleted_keys = [call.args[0] for call in mock_delete.call_args_list]
            assert f"user:{test_uid}" in deleted_keys
            assert f"auth:user:{test_uid}" in deleted_keys

    @pytest.mark.asyncio
    async def test_charge_usage_invalidates_cache_on_quota_exceeded(self):
        """Verify AgentService._charge_usage unconditionally evicts Redis cache
        even when quota limit is exceeded (atomic_increment_if_within_limit returns False)."""
        from app.models.config import AppConfigDB
        from app.services.chats import AgentService

        agent = AgentService(
            provider=MagicMock(),
            chat_repo=MagicMock(),
            user_repo=MagicMock(),
            config_repo=MagicMock(),
            attachment_service=MagicMock(),
        )
        test_uid = uuid.uuid4()
        user = _make_user(id=test_uid, tokens_used_6h=65_000)
        config = AppConfigDB()

        agent.user_repo.atomic_increment_if_within_limit = AsyncMock(return_value=False)

        with patch("app.core.redis.redis_cache.delete", new=AsyncMock()) as mock_delete:
            within_limit = await agent._charge_usage(user, 500, config)
            assert within_limit is False
            deleted_keys = [call.args[0] for call in mock_delete.call_args_list]
            assert f"user:{test_uid}" in deleted_keys
            assert f"auth:user:{test_uid}" in deleted_keys


class TestQuotaPrecheckEarlyRejection:
    """Ensures that quota exhaustion blocks generation BEFORE streaming starts,
    preventing post-generation error toasts after full responses are delivered."""

    @pytest.mark.asyncio
    async def test_quota_precheck_raises_429_immediately_when_6h_limit_exceeded(self):
        """Verify _quota_precheck raises HTTPException(429) before generation starts."""
        from fastapi import HTTPException

        from app.models.config import AppConfigDB
        from app.services.chats import AgentService

        agent = AgentService(
            provider=MagicMock(),
            chat_repo=MagicMock(),
            user_repo=MagicMock(),
            config_repo=MagicMock(),
            attachment_service=MagicMock(),
        )
        now = datetime.now(timezone.utc)
        user = _make_user(
            tokens_used_6h=30_000,
            token_limit_6h=30_000,
            reset_at=now + timedelta(hours=3),
        )
        config = AppConfigDB(default_token_limit_6h=30_000)

        with pytest.raises(HTTPException) as exc_info:
            await agent._quota_precheck(user, config)
        assert exc_info.value.status_code == 429
        assert "6-hour token limit reached" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_quota_precheck_raises_429_immediately_when_weekly_limit_exceeded(self):
        """Verify _quota_precheck raises HTTPException(429) before generation starts for weekly limit."""
        from fastapi import HTTPException

        from app.models.config import AppConfigDB
        from app.services.chats import AgentService

        agent = AgentService(
            provider=MagicMock(),
            chat_repo=MagicMock(),
            user_repo=MagicMock(),
            config_repo=MagicMock(),
            attachment_service=MagicMock(),
        )
        now = datetime.now(timezone.utc)
        user = _make_user(
            tokens_used_weekly=300_000,
            token_limit_weekly=300_000,
            weekly_reset_at=now + timedelta(days=3),
        )
        config = AppConfigDB(default_token_limit_weekly=300_000)

        with pytest.raises(HTTPException) as exc_info:
            await agent._quota_precheck(user, config)
        assert exc_info.value.status_code == 429
        assert "Weekly token limit reached" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_stream_chat_aborts_immediately_on_quota_breach_without_calling_provider(self):
        """Verify stream_chat aborts on precheck failure and does not call provider or stream response chunks."""
        from app.models.config import AppConfigDB
        from app.services.chats import AgentService

        mock_provider = MagicMock()
        mock_provider.generate_stream = AsyncMock()
        agent = AgentService(
            provider=mock_provider,
            chat_repo=MagicMock(),
            user_repo=MagicMock(),
            config_repo=MagicMock(),
            attachment_service=MagicMock(),
        )
        agent.get_active_config = AsyncMock(return_value=AppConfigDB(default_token_limit_6h=30_000))
        agent._resolve_session = AsyncMock(return_value=(MagicMock(messages=[]), uuid.uuid4(), "Test"))

        now = datetime.now(timezone.utc)
        user = _make_user(
            tokens_used_6h=30_000,
            token_limit_6h=30_000,
            reset_at=now + timedelta(hours=3),
        )

        chunks = []
        async for chunk in agent.stream_chat(user=user, message_text="Hello"):
            chunks.append(chunk)

        # Provider must NEVER be called when quota precheck fails
        mock_provider.generate_stream.assert_not_called()
        # Must yield error event immediately
        assert any("event: error" in c for c in chunks)
        assert any("6-hour token limit reached" in c for c in chunks)
