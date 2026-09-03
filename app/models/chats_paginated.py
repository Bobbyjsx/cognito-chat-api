from pydantic import BaseModel

from app.models.chats import ChatSessionSchema, MessageSchema
from app.models.pagination import PaginatedResponse


class SessionWithPaginatedMessagesSchema(BaseModel):
    session: ChatSessionSchema
    messages: PaginatedResponse[MessageSchema]
