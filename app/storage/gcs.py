"""Google Cloud Storage backend.

The ``google.cloud.storage`` import is deferred so that environments without
the dependency (or without credentials) can still use the local backend.
"""

from __future__ import annotations

import logging

from app.storage.base import StorageBackend

logger = logging.getLogger(__name__)

GCS_URI_PREFIX = "gs://"


class GCSStorageBackend(StorageBackend):
    """Stores objects in a Google Cloud Storage bucket.

    URIs are ``gs://<bucket>/<key>``.
    """

    def __init__(self, bucket_name: str, client=None):
        if not bucket_name:
            raise ValueError("GCSStorageBackend requires a bucket name (STORAGE_BUCKET).")
        from google.cloud.storage.async_client import AsyncClient

        self.bucket_name = bucket_name
        self._client = client or AsyncClient()
        self._bucket = None

    async def _get_bucket(self):
        if self._bucket is None:
            self._bucket = await self._client.bucket(self.bucket_name)
        return self._bucket

    @staticmethod
    def _key_from_uri(uri: str) -> str:
        if uri.startswith(GCS_URI_PREFIX):
            return uri[len(GCS_URI_PREFIX):].split("/", 1)[1]
        return uri

    async def upload_bytes(self, key: str, data: bytes, content_type: str) -> str:
        bucket = await self._get_bucket()
        blob = bucket.blob(key)
        await blob.upload_from_string(data, content_type=content_type)
        logger.info("Uploaded object gs://%s/%s", self.bucket_name, key)
        return f"{GCS_URI_PREFIX}{self.bucket_name}/{key}"

    async def read_bytes(self, uri: str) -> bytes:
        bucket = await self._get_bucket()
        blob = bucket.blob(self._key_from_uri(uri))
        return await blob.download_as_bytes()

    async def delete(self, uri: str) -> None:
        bucket = await self._get_bucket()
        blob = bucket.blob(self._key_from_uri(uri))
        await blob.delete()
