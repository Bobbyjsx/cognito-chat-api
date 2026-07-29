"""Tests for 6-hourly and weekly token quota enforcement.

All time-dependent tests freeze `datetime.now` via `freezegun` so we can
simulate window expiry without actually waiting.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from freezegun import freeze_time

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
    client.post("/auth/signup", json={"email": email, "password": password})
    resp = client.post("/auth/login", json={"email": email, "password": password})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


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

    def test_6h_token_limit_default_is_60k(self):
        user = UserDB(email="a@b.com", hashed_password="x")
        assert user.token_limit_6h == 60_000

    def test_weekly_token_limit_default_is_300k(self):
        user = UserDB(email="a@b.com", hashed_password="x")
        assert user.token_limit_weekly == 300_000


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
            "tokens_used_6h": 59_999,   # Would normally block, but window is expired
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
        headers = _auth_headers(client, "quota6h@example.com")

        now = datetime.now(timezone.utc)
        exhausted_user = _make_user(
            email="quota6h@example.com",
            tokens_used_6h=60_000,
            token_limit_6h=60_000,
            reset_at=now + timedelta(hours=6),
        )

        with patch("app.api.dependencies.UserRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=exhausted_user)
            resp = client.post("/agent/chat", headers=headers, json={"message": "hi"})

        assert resp.status_code == 429
        assert "6-hour" in resp.json()["detail"]

    def test_chat_blocked_when_weekly_limit_reached(self, client, mock_agent):
        headers = _auth_headers(client, "quotawk@example.com")

        now = datetime.now(timezone.utc)
        exhausted_user = _make_user(
            email="quotawk@example.com",
            tokens_used_weekly=300_000,
            token_limit_weekly=300_000,
            weekly_reset_at=now + timedelta(weeks=1),
        )

        with patch("app.api.dependencies.UserRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=exhausted_user)
            resp = client.post("/agent/chat", headers=headers, json={"message": "hi"})

        assert resp.status_code == 429
        assert "Weekly" in resp.json()["detail"]

    def test_chat_allowed_after_6h_window_expires(self, client, mock_agent):
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
        ):
            MockAuthRepo.return_value.get_by_id = AsyncMock(return_value=user_past_reset)
            # AgentService uses a separate UserRepository instance for quota increment
            MockChatUserRepo.return_value.atomic_increment_if_within_limit = AsyncMock(return_value=True)
            resp = client.post("/agent/chat", headers=headers, json={"message": "hi"})

        # Pre-gen check should pass (window expired → effective_6h = 0)
        assert resp.status_code == 200

    def test_me_exposes_quota_fields(self, client):
        headers = _auth_headers(client, "quotame@example.com")
        resp = client.get("/auth/me", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens_used_6h" in data
        assert "token_limit_6h" in data
        assert "reset_at" in data
        assert "tokens_used_weekly" in data
        assert "token_limit_weekly" in data
        assert "weekly_reset_at" in data
        assert data["token_limit_6h"] == 60_000
        assert data["token_limit_weekly"] == 300_000

    @freeze_time("2026-07-29 06:00:00", tz_offset=0)
    def test_reset_at_is_6h_from_signup(self, client):
        headers = _auth_headers(client, "quotareset@example.com")
        me = client.get("/auth/me", headers=headers).json()
        reset_at = datetime.fromisoformat(me["reset_at"])
        expected = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
        assert reset_at == expected

    @freeze_time("2026-07-29 06:00:00", tz_offset=0)
    def test_weekly_reset_at_is_7_days_from_signup(self, client):
        headers = _auth_headers(client, "quotawkreset@example.com")
        me = client.get("/auth/me", headers=headers).json()
        weekly_reset_at = datetime.fromisoformat(me["weekly_reset_at"])
        expected = datetime(2026, 8, 5, 6, 0, 0, tzinfo=timezone.utc)
        assert weekly_reset_at == expected

    def test_reset_timestamp_is_persistent_across_multiple_requests(self, client):
        headers = _auth_headers(client, "persist@example.com")
        res1 = client.get("/auth/me", headers=headers).json()
        res2 = client.get("/auth/me", headers=headers).json()
        res3 = client.get("/auth/me", headers=headers).json()

        assert res1["reset_at"] == res2["reset_at"] == res3["reset_at"]
        assert res1["weekly_reset_at"] == res2["weekly_reset_at"] == res3["weekly_reset_at"]

