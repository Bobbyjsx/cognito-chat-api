from app.models.pagination import PaginatedResponse
from app.models.chats import ChatSessionListSchema
import uuid
import datetime

item = ChatSessionListSchema(id=uuid.uuid4(), user_id=uuid.uuid4(), title="Test", created_at=datetime.datetime.now(), updated_at=datetime.datetime.now(), read_status="read")
resp = PaginatedResponse[ChatSessionListSchema](items=[item], limit=10, offset=0, has_more=False)
print(resp.model_dump_json())
