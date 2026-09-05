import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from google.cloud import firestore
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.models.chats import ChatMessageDB, ChatSessionDB, MessageRole, ReadStatus, clip_session_preview

logger = logging.getLogger(__name__)

# Extra docs to absorb soft-deleted rows that still sit in the ordered index.
_LIST_DELETED_SLACK = 8


class ChatRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("sessions")

    async def create_session(self, user_id: UUID | str, title: str | None = None) -> ChatSessionDB:
        session_db = ChatSessionDB(user_id=user_id, title=title)
        doc_ref = self.collection.document(str(session_db.id))
        data = session_db.model_dump(mode="json")
        await doc_ref.set(data)
        return session_db

    async def update_session_title(self, session_id: UUID | str, title: str) -> None:
        doc_ref = self.collection.document(str(session_id))
        await doc_ref.update({"title": title})

    async def session_exists(self, session_id: UUID | str, user_id: UUID | str) -> bool:
        doc_ref = self.collection.document(str(session_id))
        doc = await doc_ref.get()
        if not doc.exists:
            return False
        data = doc.to_dict() or {}
        if data.get("user_id") != str(user_id):
            return False
        return data.get("is_deleted") is False

    async def get_session(
        self, session_id: UUID | str, user_id: UUID | str, limit: int = 10, offset: int = 0
    ) -> tuple[ChatSessionDB | None, bool]:
        doc_ref = self.collection.document(str(session_id))

        messages_query = (
            doc_ref.collection("messages")
            .order_by("created_at", direction=firestore.Query.ASCENDING)
            .offset(offset)
            .limit(limit + 1)
        )

        async def _fetch_messages():
            msgs = []
            async for msg_doc in messages_query.stream():
                msg_data = msg_doc.to_dict() or {}
                if msg_data:
                    msgs.append(ChatMessageDB(**msg_data))
            return msgs

        # Concurrently fetch parent session document and messages subcollection
        doc, msgs = await asyncio.gather(doc_ref.get(), _fetch_messages())

        if not doc.exists:
            return None, False

        data = doc.to_dict() or {}
        if not data or data.get("user_id") != str(user_id) or data.get("is_deleted") is True:
            return None, False

        session = ChatSessionDB(**data)

        has_more = len(msgs) > limit
        if has_more:
            msgs.pop()

        session.messages = msgs

        return session, has_more

    def _session_from_doc(self, data: dict[str, Any] | None) -> ChatSessionDB | None:
        if not data or data.get("is_deleted") is True:
            return None
        return ChatSessionDB(**data)

    async def get_user_sessions(
        self, user_id: UUID | str, search_query: str | None = None, limit: int = 10, offset: int = 0
    ) -> tuple[list[ChatSessionDB], bool, int]:
        limit = max(1, limit)
        offset = max(0, offset)
        q_lower = search_query.strip().lower() if search_query and search_query.strip() else None
        if q_lower:
            return await self._search_user_sessions(user_id, q_lower, limit, offset)
        return await self._list_user_sessions(user_id, limit, offset)

    async def _list_user_sessions(
        self, user_id: UUID | str, limit: int, offset: int
    ) -> tuple[list[ChatSessionDB], bool, int]:
        query = (
            self.collection.where(filter=FieldFilter("user_id", "==", str(user_id)))
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(offset + limit + 1 + _LIST_DELETED_SLACK)
        )
        sessions: list[ChatSessionDB] = []
        skipped = 0
        async for doc in query.stream():
            session = self._session_from_doc(doc.to_dict())
            if session is None:
                continue
            if skipped < offset:
                skipped += 1
                continue
            sessions.append(session)
            if len(sessions) > limit:
                break

        has_more = len(sessions) > limit
        if has_more:
            sessions = sessions[:limit]
        total = offset + len(sessions) + (1 if has_more else 0)
        return sessions, has_more, total

    async def _search_user_sessions(
        self, user_id: UUID | str, q_lower: str, limit: int, offset: int
    ) -> tuple[list[ChatSessionDB], bool, int]:
        docs = (
            self.collection.where(filter=FieldFilter("user_id", "==", str(user_id)))
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .stream()
        )
        sessions: list[ChatSessionDB] = []
        async for doc in docs:
            session = self._session_from_doc(doc.to_dict())
            if session is None:
                continue

            title_match = bool(session.title and q_lower in session.title.lower())
            content_match = bool(session.last_message_content and q_lower in session.last_message_content.lower())
            if not (title_match or content_match):
                messages_ref = self.collection.document(str(session.id)).collection("messages")
                deep_match = False
                async for msg_doc in messages_ref.stream():
                    msg_data = msg_doc.to_dict() or {}
                    msg_content = msg_data.get("content", "")
                    if msg_content and q_lower in msg_content.lower():
                        deep_match = True
                        break
                if not deep_match:
                    continue

            sessions.append(session)

        total = len(sessions)
        paginated_sessions = sessions[offset : offset + limit]
        has_more = offset + limit < total
        return paginated_sessions, has_more, total

    async def soft_delete_session(self, session_id: UUID | str, user_id: UUID | str) -> bool:
        try:
            doc_ref = self.collection.document(str(session_id))
            doc = await doc_ref.get()
            if not doc.exists:
                return False

            data = doc.to_dict() or {}
            if not data or data.get("user_id") != str(user_id):
                return False

            await doc_ref.update(
                {
                    "is_deleted": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            return True
        except Exception:
            logger.exception("Error soft-deleting session %s", session_id)
            return False

    async def add_message(
        self,
        session_id: UUID | str,
        role: MessageRole | str,
        content: str,
        error: str | None = None,
        attachment_ids: list[str] | None = None,
        generation_id: str | UUID | None = None,
        created_at: datetime | str | None = None,
        parts: list[dict[str, Any]] | None = None,
        update_session_summary: bool = True,
    ) -> ChatMessageDB:
        # Convert generation_id to UUID if it's a string
        if isinstance(generation_id, str):
            generation_id = UUID(generation_id)

        clean_parts = []
        for p in parts or []:
            if isinstance(p, dict) and p.get("type") == "file":
                cleaned = {k: v for k, v in p.items() if k not in ("url", "url_expires_at", "urlExpiresAt")}
                clean_parts.append(cleaned)
            else:
                clean_parts.append(p)

        msg_kwargs = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "error": error,
            "attachment_ids": attachment_ids or [],
            "generation_id": generation_id,
            "parts": clean_parts,
        }
        if created_at is not None:
            msg_kwargs["created_at"] = created_at

        message_db = ChatMessageDB(**msg_kwargs)
        doc_ref = self.collection.document(str(session_id)).collection("messages").document(str(message_db.id))
        data = message_db.model_dump(mode="json")

        role_val = role.value if isinstance(role, MessageRole) else str(role)
        read_status = ReadStatus.READ if role_val == MessageRole.USER.value else ReadStatus.NOT_READ
        session_ref = self.collection.document(str(session_id))

        batch = self.db.batch()
        batch.set(doc_ref, data)
        if update_session_summary:
            batch.update(
                session_ref,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "last_message_content": clip_session_preview(content or (error or "Error responding")),
                    "last_message_role": role_val,
                    "read_status": read_status.value,
                },
            )
        await batch.commit()

        return message_db

    async def mark_session_read(self, session_id: UUID | str) -> None:
        session_ref = self.collection.document(str(session_id))
        await session_ref.update({"read_status": ReadStatus.READ.value})

    async def update_message(self, session_id: UUID | str, message_id: UUID | str, **fields) -> None:
        doc_ref = self.collection.document(str(session_id)).collection("messages").document(str(message_id))
        await doc_ref.update(fields)
