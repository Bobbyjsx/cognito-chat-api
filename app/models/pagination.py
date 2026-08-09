from typing import TypeVar, Generic
from pydantic import BaseModel

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int | None = None
    limit: int
    offset: int
    has_more: bool
