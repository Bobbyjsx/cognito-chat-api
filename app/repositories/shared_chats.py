import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.core.cache_keys import CacheKeys
from app.core.redis import redis_cache
from app.models.chats import (
    ChatSessionDB,
    MessageRole,
    ReadStatus,
    SharedChatDB,
    SharedChatMessageDB,
    clip_session_preview,
)

logger = logging.getLogger(__name__)


class SharedChatRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.sessions_collection = self.db.collection("sessions")
        self.shared_collection = self.db.collection("shared_chats")

    async def create_or_update_share(
        self,
        session_id: UUID | str,
        user_id: UUID | str,
        title: str | None = None,
        show_name: bool = True,
        author_name: str | None = None,
    ) -> SharedChatDB | None:
        """Freezes and creates/updates a shared chat snapshot up to the current moment.

        Strictly filters out system/developer/tool messages and scrubs system preambles.
        """
        session_ref = self.sessions_collection.document(str(session_id))
        now = datetime.now(timezone.utc)

        async def _fetch_messages():
            msgs = []
            async for doc in session_ref.collection("messages").stream():
                data = doc.to_dict() or {}
                if data:
                    msgs.append(data)
            return msgs

        async def _fetch_existing_share():
            async for s_doc in (
                self.shared_collection.where(filter=FieldFilter("session_id", "==", str(session_id)))
                .where(filter=FieldFilter("user_id", "==", str(user_id)))
                .limit(1)
                .stream()
            ):
                return s_doc.id, s_doc.to_dict() or {}
            return None, {}

        # Concurrently fetch session metadata, raw messages, and any existing share document
        session_doc, raw_msgs, (existing_share_id, existing_share_dict) = await asyncio.gather(
            session_ref.get(),
            _fetch_messages(),
            _fetch_existing_share(),
        )

        if not session_doc.exists:
            return None

        session_data = session_doc.to_dict() or {}
        if str(session_data.get("user_id")) != str(user_id) or session_data.get("is_deleted") is True:
            return None

        # Sort messages in Python memory by created_at
        def _parse_created_at(item: dict) -> datetime:
            val = item.get("created_at")
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except Exception:
                    logger.error(f"Invalid created_at value: {val}")
            return now

        raw_msgs.sort(key=_parse_created_at)

        # Pre-fetch attachment metadata for any attached files
        all_att_ids = set()
        for msg_data in raw_msgs:
            for aid in msg_data.get("attachment_ids", []) or []:
                all_att_ids.add(str(aid))

        att_map = {}
        if all_att_ids:
            from app.repositories.attachments import AttachmentRepository

            att_repo = AttachmentRepository(self.db)
            att_metas = await att_repo.get_many(user_id, list(all_att_ids))
            att_map = {str(m.id): m for m in att_metas}

        snapshot_messages: list[SharedChatMessageDB] = []
        for msg_data in raw_msgs:
            raw_role = msg_data.get("role")
            role_val = raw_role.value if isinstance(raw_role, MessageRole) else str(raw_role)
            # Filter out internal messages: only include messages where role is "user" or "agent"
            if role_val not in ("user", "agent", MessageRole.USER.value, MessageRole.AGENT.value):
                continue

            # Scrub system instructions, developer preambles, and tool execution traces from parts
            clean_parts = []
            for part in msg_data.get("parts", []):
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", "")).lower()
                if part_type in (
                    "system",
                    "developer",
                    "system_instruction",
                    "instruction",
                    "tool",
                    "tool_call",
                    "tool_result",
                    "internal",
                ):
                    continue
                part_text = str(part.get("text", ""))
                if "You are Cognito" in part_text and "Security & Confidentiality Guardrails" in part_text:
                    continue
                if isinstance(part, dict) and part.get("type") == "file":
                    part = {k: v for k, v in part.items() if k not in ("url", "url_expires_at", "urlExpiresAt")}
                clean_parts.append(part)

            content = msg_data.get("content", "")
            if "You are Cognito" in content and "Security & Confidentiality Guardrails" in content:
                continue

            # Ensure file parts are present in clean_parts if message references attachment_ids
            has_file = any(p.get("type") == "file" for p in clean_parts if isinstance(p, dict))
            if not has_file and msg_data.get("attachment_ids"):
                for aid in msg_data["attachment_ids"]:
                    s_aid = str(aid)
                    if s_aid in att_map:
                        meta = att_map[s_aid]
                        clean_parts.append(
                            {
                                "type": "file",
                                "attachment_id": str(meta.id),
                                "filename": meta.filename,
                                "contentType": meta.mime_type,
                                "mediaType": meta.mime_type,
                                "size": meta.size,
                                "bucket": meta.bucket,
                                "object_name": meta.object_name,
                                "storage_uri": meta.storage_uri,
                            }
                        )

            snapshot_messages.append(
                SharedChatMessageDB(
                    id=msg_data.get("id") or uuid4(),
                    role=MessageRole.USER if role_val == "user" else MessageRole.AGENT,
                    content=content,
                    attachment_ids=msg_data.get("attachment_ids", []),
                    parts=clean_parts,
                    created_at=msg_data.get("created_at") or now,
                )
            )

        existing_created_at = None
        if existing_share_dict:
            c_at = existing_share_dict.get("created_at")
            if isinstance(c_at, datetime):
                existing_created_at = c_at
            elif isinstance(c_at, str):
                try:
                    existing_created_at = datetime.fromisoformat(c_at)
                except Exception:
                    existing_created_at = now

        share_id = existing_share_id or uuid4().hex[:12]
        resolved_title = title or session_data.get("title") or "Shared Chat"

        shared_chat = SharedChatDB(
            id=share_id,
            session_id=UUID(str(session_id)),
            user_id=str(user_id),
            title=resolved_title,
            show_name=show_name,
            author_name=author_name if show_name else "Anonymous",
            revoked_at=None,
            created_at=existing_created_at or now,
            updated_at=now,
            message_count=len(snapshot_messages),
            messages=snapshot_messages,
        )

        # Persist to Firestore and pre-warm Redis cache concurrently, appending share_id to parent session
        doc_ref = self.shared_collection.document(share_id)
        dumped_data = shared_chat.model_dump(mode="json")
        await asyncio.gather(
            doc_ref.set(dumped_data),
            session_ref.update(
                {
                    "share_id": share_id,
                    "updated_at": now.isoformat(),
                }
            ),
            redis_cache.set(CacheKeys.shared_chat(share_id), dumped_data, expire=300),
            redis_cache.set(CacheKeys.session_share(session_id), dumped_data, expire=300),
            redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id)),
            redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user_id)),
        )

        return shared_chat

    async def get_share_by_session(self, session_id: UUID | str, user_id: UUID | str) -> SharedChatDB | None:
        """Retrieves active (non-revoked) share snapshot for a specific session."""
        cache_key = CacheKeys.session_share(session_id)
        cached = await redis_cache.get(cache_key)
        if cached:
            try:
                share_obj = SharedChatDB(**cached)
                if str(share_obj.user_id) == str(user_id) and share_obj.revoked_at is None:
                    return share_obj
            except Exception:
                logger.warning("Failed to deserialize cached session share %s", session_id)

        async for s_doc in (
            self.shared_collection.where(filter=FieldFilter("session_id", "==", str(session_id)))
            .where(filter=FieldFilter("user_id", "==", str(user_id)))
            .limit(1)
            .stream()
        ):
            data = s_doc.to_dict() or {}
            if data and not data.get("revoked_at"):
                shared_chat = SharedChatDB(**data)
                await redis_cache.set(cache_key, shared_chat.model_dump(mode="json"), expire=300)
                return shared_chat

        return None

    async def get_shared_chat(self, share_id: str) -> SharedChatDB | None:
        """Retrieves a shared chat snapshot, utilizing Redis caching for active links."""
        cache_key = CacheKeys.shared_chat(share_id)
        cached = await redis_cache.get(cache_key)
        if cached:
            try:
                return SharedChatDB(**cached)
            except Exception:
                logger.warning("Failed to deserialize cached shared chat %s", share_id)

        doc_ref = self.shared_collection.document(share_id)
        doc = await doc_ref.get()
        if not doc.exists:
            return None

        data = doc.to_dict() or {}
        shared_chat = SharedChatDB(**data)

        # Cache in Redis only if not revoked
        if shared_chat.revoked_at is None:
            await redis_cache.set(cache_key, shared_chat.model_dump(mode="json"), expire=300)
        return shared_chat

    async def revoke_shared_chat(self, share_id: str, user_id: UUID | str) -> bool:
        """Revokes a public shared chat link, purging the cache."""
        doc_ref = self.shared_collection.document(share_id)
        doc = await doc_ref.get()
        if not doc.exists:
            return False

        data = doc.to_dict() or {}
        if str(data.get("user_id")) != str(user_id):
            return False

        now = datetime.now(timezone.utc)
        update_tasks = [
            doc_ref.update(
                {
                    "revoked_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                }
            )
        ]
        session_id = data.get("session_id")
        delete_tasks = [redis_cache.delete(CacheKeys.shared_chat(share_id))]
        if session_id:
            update_tasks.append(
                self.sessions_collection.document(str(session_id)).update(
                    {
                        "share_id": None,
                        "updated_at": now.isoformat(),
                    }
                )
            )
            delete_tasks.append(redis_cache.delete(CacheKeys.session_share(session_id)))
            delete_tasks.append(redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id)))
            delete_tasks.append(redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user_id)))
        await asyncio.gather(*update_tasks, *delete_tasks)
        return True

    async def revoke_session_share(self, session_id: UUID | str, user_id: UUID | str) -> bool:
        """Revokes any active public share links associated with a session."""
        share_query = (
            self.shared_collection.where(filter=FieldFilter("session_id", "==", str(session_id)))
            .where(filter=FieldFilter("user_id", "==", str(user_id)))
            .stream()
        )
        revoked_any = False
        now = datetime.now(timezone.utc)
        update_tasks = [
            self.sessions_collection.document(str(session_id)).update(
                {
                    "share_id": None,
                    "updated_at": now.isoformat(),
                }
            )
        ]
        delete_tasks = [
            redis_cache.delete(CacheKeys.session_share(session_id)),
            redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id)),
            redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user_id)),
        ]
        async for s_doc in share_query:
            share_id = s_doc.id
            update_tasks.append(
                self.shared_collection.document(share_id).update(
                    {
                        "revoked_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }
                )
            )
            delete_tasks.append(redis_cache.delete(CacheKeys.shared_chat(share_id)))
            revoked_any = True

        if revoked_any:
            await asyncio.gather(*update_tasks, *delete_tasks)
        return revoked_any

    async def continue_shared_chat(self, share_id: str, target_user_id: UUID | str) -> ChatSessionDB | None:
        """Clones a shared chat snapshot into a new private session with exclude_from_memory=True."""
        shared_chat = await self.get_shared_chat(share_id)
        if not shared_chat or shared_chat.revoked_at is not None:
            return None

        messages = shared_chat.messages or []
        last_content = None
        last_role = None
        if messages:
            last_msg = messages[-1]
            last_content = last_msg.content
            last_role = last_msg.role.value if isinstance(last_msg.role, MessageRole) else str(last_msg.role)

        new_session = ChatSessionDB(
            user_id=str(target_user_id),
            title=shared_chat.title or "Continued Chat",
            exclude_from_memory=True,
            last_message_content=clip_session_preview(last_content),
            last_message_role=last_role,
            read_status=ReadStatus.READ,
        )
        new_session_ref = self.sessions_collection.document(str(new_session.id))

        batch = self.db.batch()
        batch.set(new_session_ref, new_session.model_dump(mode="json"))

        for msg in messages:
            msg_id = uuid4()
            msg_doc_ref = new_session_ref.collection("messages").document(str(msg_id))
            role_val = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role)
            created_at_val = msg.created_at.isoformat() if isinstance(msg.created_at, datetime) else str(msg.created_at)
            msg_data = {
                "id": str(msg_id),
                "session_id": str(new_session.id),
                "role": role_val,
                "content": msg.content,
                "attachment_ids": msg.attachment_ids,
                "parts": msg.parts,
                "created_at": created_at_val,
            }
            batch.set(msg_doc_ref, msg_data)

        await batch.commit()

        # Invalidate target user's sessions list cache in Redis
        await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(target_user_id))

        new_session.last_message_content = clip_session_preview(last_content)
        new_session.last_message_role = last_role
        return new_session
