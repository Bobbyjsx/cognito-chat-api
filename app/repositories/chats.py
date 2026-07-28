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
        doc_ref = self.collection.document(str(session_id))
        doc = await doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            if data.get("user_id") != str(user_id):
                return None

            session = ChatSessionDB(**data)

            # Fetch messages subcollection
            messages_ref = doc_ref.collection("messages").order_by("created_at").stream()
            async for msg_doc in messages_ref:
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
        await doc_ref.set(data)

        # Update session updated_at
        from datetime import datetime, timezone

        session_ref = self.collection.document(str(session_id))
        await session_ref.update({"updated_at": datetime.now(timezone.utc).isoformat()})

        return message_db
