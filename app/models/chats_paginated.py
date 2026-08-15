from pydantic import BaseModel

from app.models.chats import ChatMessageSchema, ChatSessionSchema
from app.models.pagination import PaginatedResponse


class SessionWithPaginatedMessagesSchema(BaseModel):
    session: ChatSessionSchema
    messages: PaginatedResponse[ChatMessageSchema]
