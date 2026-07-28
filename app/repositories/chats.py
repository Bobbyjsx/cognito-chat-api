from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.models.chats import ChatMessageDB, ChatSessionDB


class ChatRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("sessions")

    async def create_session(self, user_id: UUID) -> ChatSessionDB:
        session_db = ChatSessionDB(user_id=user_id)
        doc_ref = self.collection.document(str(session_db.id))
        data = session_db.model_dump(mode="json")
        await doc_ref.set(data)
        return session_db

    async def get_session(self, session_id: UUID, user_id: UUID) -> ChatSessionDB | None:
        import asyncio

        doc_ref = self.collection.document(str(session_id))
        messages_ref = doc_ref.collection("messages").order_by("created_at")

        # Fetch session document and messages subcollection concurrently
        doc, messages_docs = await asyncio.gather(doc_ref.get(), messages_ref.get())

        if doc.exists:
            data = doc.to_dict()
            if data.get("user_id") != str(user_id):
                return None

            session = ChatSessionDB(**data)

            for msg_doc in messages_docs:
                msg_data = msg_doc.to_dict()
                session.messages.append(ChatMessageDB(**msg_data))

            return session
        return None

    async def get_user_sessions(self, user_id: UUID) -> list[ChatSessionDB]:
        sessions = []
        docs = (
            self.collection.where(filter=FieldFilter("user_id", "==", str(user_id)))
            .order_by("updated_at", direction="DESCENDING")
            .stream()
        )
        async for doc in docs:
            data = doc.to_dict()
            session = ChatSessionDB(**data)
            sessions.append(session)
        return sessions

    async def add_message(self, session_id: UUID, role: str, content: str) -> ChatMessageDB:
        message_db = ChatMessageDB(session_id=session_id, role=role, content=content)
        doc_ref = self.collection.document(str(session_id)).collection("messages").document(str(message_db.id))
        data = message_db.model_dump(mode="json")

        from datetime import datetime, timezone

        read_status = "read" if role == "user" else "not read"

        session_ref = self.collection.document(str(session_id))

        # Use batch write to cut network roundtrips in half
        batch = self.db.batch()
        batch.set(doc_ref, data)
        batch.update(
            session_ref,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_message_content": content,
                "last_message_role": role,
                "read_status": read_status,
            },
        )
        await batch.commit()

        return message_db

    async def mark_session_read(self, session_id: UUID) -> None:
        session_ref = self.collection.document(str(session_id))
        await session_ref.update({"read_status": "read"})
