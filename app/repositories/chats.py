import logging
from datetime import datetime, timezone
from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.models.chats import ChatMessageDB, ChatSessionDB

logger = logging.getLogger(__name__)


class ChatRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("sessions")

    async def create_session(self, user_id: UUID, title: str | None = None) -> ChatSessionDB:
        session_db = ChatSessionDB(user_id=user_id, title=title)
        doc_ref = self.collection.document(str(session_db.id))
        data = session_db.model_dump(mode="json")
        await doc_ref.set(data)
        return session_db

    async def update_session_title(self, session_id: UUID, title: str) -> None:
        doc_ref = self.collection.document(str(session_id))
        await doc_ref.update({"title": title})

    async def session_exists(self, session_id: UUID, user_id: UUID) -> bool:
        doc_ref = self.collection.document(str(session_id))
        doc = await doc_ref.get()
        if not doc.exists:
            return False
        data = doc.to_dict()
        if data.get("user_id") != str(user_id):
            return False
        if data.get("is_deleted") is True:
            return False
        return True

    async def get_session(self, session_id: UUID, user_id: UUID, limit: int = 10, offset: int = 0) -> tuple[ChatSessionDB | None, bool]:
        doc_ref = self.collection.document(str(session_id))
        from google.cloud import firestore
        messages_ref = doc_ref.collection("messages").order_by("created_at", direction=firestore.Query.DESCENDING).offset(offset).limit(limit + 1)

        doc = await doc_ref.get()
        if not doc.exists:
            return None, False

        data = doc.to_dict()
        if data.get("user_id") != str(user_id):
            return None, False
        if data.get("is_deleted") is True:
            return None, False

        session = ChatSessionDB(**data)

        msgs = []
        async for msg_doc in messages_ref.stream():
            msg_data = msg_doc.to_dict()
            msgs.append(ChatMessageDB(**msg_data))

        has_more = len(msgs) > limit
        if has_more:
            msgs.pop()

        # Reverse back to chronological order
        msgs.reverse()
        session.messages = msgs

        return session, has_more

    async def get_user_sessions(self, user_id: UUID, search_query: str | None = None, limit: int = 10, offset: int = 0) -> tuple[list[ChatSessionDB], bool, int]:
        sessions = []
        docs = self.collection.where(filter=FieldFilter("user_id", "==", str(user_id))).stream()
        q_lower = search_query.strip().lower() if search_query and search_query.strip() else None

        async for doc in docs:
            data = doc.to_dict()
            if data.get("is_deleted") is True:
                continue

            session = ChatSessionDB(**data)

            if q_lower:
                title_match = bool(session.title and q_lower in session.title.lower())
                content_match = bool(session.last_message_content and q_lower in session.last_message_content.lower())

                if not (title_match or content_match):
                    # Deep search message history using async stream
                    messages_ref = self.collection.document(str(session.id)).collection("messages")
                    deep_match = False
                    async for msg_doc in messages_ref.stream():
                        msg_data = msg_doc.to_dict()
                        msg_content = msg_data.get("content", "")
                        if msg_content and q_lower in msg_content.lower():
                            deep_match = True
                            break
                    if not deep_match:
                        continue

            sessions.append(session)

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        total = len(sessions)
        paginated_sessions = sessions[offset : offset + limit]
        has_more = offset + limit < total
        return paginated_sessions, has_more, total

    async def soft_delete_session(self, session_id: UUID, user_id: UUID) -> bool:
        try:
            doc_ref = self.collection.document(str(session_id))
            doc = await doc_ref.get()
            if not doc.exists:
                return False

            data = doc.to_dict()
            if data.get("user_id") != str(user_id):
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
        session_id: UUID,
        role: str,
        content: str,
        error: str | None = None,
        attachment_ids: list[str] | None = None,
    ) -> ChatMessageDB:
        message_db = ChatMessageDB(
            session_id=session_id,
            role=role,
            content=content,
            error=error,
            attachment_ids=attachment_ids or [],
        )
        doc_ref = self.collection.document(str(session_id)).collection("messages").document(str(message_db.id))
        data = message_db.model_dump(mode="json")

        read_status = "read" if role == "user" else "not read"
        session_ref = self.collection.document(str(session_id))

        batch = self.db.batch()
        batch.set(doc_ref, data)
        batch.update(
            session_ref,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_message_content": content or (error or "Error responding"),
                "last_message_role": role,
                "read_status": read_status,
            },
        )
        await batch.commit()

        return message_db

    async def mark_session_read(self, session_id: UUID) -> None:
        session_ref = self.collection.document(str(session_id))
        await session_ref.update({"read_status": "read"})
