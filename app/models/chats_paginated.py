from pydantic import BaseModel
from app.models.chats import ChatSessionSchema, ChatMessageSchema
from app.models.pagination import PaginatedResponse

class SessionWithPaginatedMessagesSchema(BaseModel):
    session: ChatSessionSchema
    messages: PaginatedResponse[ChatMessageSchema]
