"""Abstract object storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Minimal object storage interface used by the attachment pipeline."""

    @abstractmethod
    async def upload_bytes(self, key: str, data: bytes, content_type: str) -> str:
        """Store ``data`` under ``key`` and return a URI that ``read_bytes`` and
        ``delete`` accept."""

    @abstractmethod
    async def read_bytes(self, uri: str) -> bytes:
        """Return the bytes previously stored at ``uri``."""

    @abstractmethod
    async def delete(self, uri: str) -> None:
        """Remove the object at ``uri`` (no-op if it does not exist)."""

    @abstractmethod
    async def move(self, old_uri: str, new_key: str) -> str:
        """Move the object at ``old_uri`` to ``new_key`` and return the new URI."""
