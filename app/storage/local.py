"""Filesystem-backed storage backend for development and tests."""

from __future__ import annotations

import logging
from pathlib import Path

from app.storage.base import StorageBackend

logger = logging.getLogger(__name__)

LOCAL_URI_PREFIX = "local://"


class LocalStorageBackend(StorageBackend):
    """Stores objects under a local directory.

    URIs look like ``local://<relative-path>``. Not intended for production.
    """

    def __init__(self, root: str):
        self.root = Path(root)

    def _path_for(self, uri: str) -> Path:
        if uri.startswith(LOCAL_URI_PREFIX):
            rel = uri[len(LOCAL_URI_PREFIX) :]
        else:
            rel = uri
        path = (self.root / rel).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError(f"Invalid storage URI: {uri!r}")
        return path

    async def upload_bytes(self, key: str, data: bytes, content_type: str) -> str:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"{LOCAL_URI_PREFIX}{key}"

    async def read_bytes(self, uri: str) -> bytes:
        path = self._path_for(uri)
        if not path.exists():
            raise FileNotFoundError(f"Object not found: {uri}")
        return path.read_bytes()

    async def delete(self, uri: str) -> None:
        path = self._path_for(uri)
        if path.exists():
            path.unlink()
            logger.info("Deleted local object %s", uri)

    async def move(self, old_uri: str, new_key: str) -> str:
        old_path = self._path_for(old_uri)
        if not old_path.exists():
            raise FileNotFoundError(f"Object not found: {old_uri}")
        new_path = self._path_for(new_key)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
        logger.info("Moved local object %s to %s", old_uri, new_key)
        return f"{LOCAL_URI_PREFIX}{new_key}"
