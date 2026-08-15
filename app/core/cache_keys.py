import uuid


class CacheKeys:
    @staticmethod
    def user_sessions(user_id: uuid.UUID | str, limit: int, offset: int, search_query: str | None = None) -> str:
        q = search_query.strip().lower() if search_query and search_query.strip() else ""
        return f"sessions:{user_id}:limit={limit}:offset={offset}:q={q}"

    @staticmethod
    def user_attachments(user_id: uuid.UUID | str, limit: int, offset: int, session_id: uuid.UUID | str | None = None, type_filter: str | None = None, search_query: str | None = None) -> str:
        s = session_id or "all"
        t = type_filter or "all"
        q = search_query.strip().lower() if search_query and search_query.strip() else ""
        return f"attachments:{user_id}:session={s}:type={t}:q={q}:limit={limit}:offset={offset}"

    @staticmethod
    def session_details(session_id: uuid.UUID | str, limit: int, offset: int) -> str:
        return f"session:{session_id}:limit={limit}:offset={offset}"

    @staticmethod
    def system_config() -> str:
        return "config:system"

    @staticmethod
    def user_profile(user_id: uuid.UUID | str) -> str:
        return f"user:{user_id}"
