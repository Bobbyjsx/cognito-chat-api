"""Google Cloud Storage backend.

The ``google.cloud.storage`` import is deferred so that environments without
the dependency (or without credentials) can still use the local backend.
"""

from __future__ import annotations

import asyncio
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
        from google.cloud.storage import Client

        self.bucket_name = bucket_name
        self._client = client or Client()
        self._bucket = self._client.bucket(self.bucket_name)

    @staticmethod
    def _key_from_uri(uri: str) -> str:
        if uri.startswith(GCS_URI_PREFIX):
            return uri[len(GCS_URI_PREFIX):].split("/", 1)[1]
        return uri

    async def upload_bytes(self, key: str, data: bytes, content_type: str) -> str:
        def _upload():
            blob = self._bucket.blob(key)
            blob.upload_from_string(data, content_type=content_type)
        await asyncio.to_thread(_upload)
        logger.info("Uploaded object gs://%s/%s", self.bucket_name, key)
        return f"{GCS_URI_PREFIX}{self.bucket_name}/{key}"

    async def read_bytes(self, uri: str) -> bytes:
        def _download():
            from google.cloud.exceptions import NotFound
            blob = self._bucket.blob(self._key_from_uri(uri))
            try:
                return blob.download_as_bytes()
            except NotFound:
                # Fallback to permanent path if temp fails
                if "/temp/" in uri:
                    perm_uri = uri.replace("/temp/", "/")
                    perm_blob = self._bucket.blob(self._key_from_uri(perm_uri))
                    try:
                        return perm_blob.download_as_bytes()
                    except NotFound:
                        pass
                raise ValueError("Object not found in GCS")
                
        try:
            return await asyncio.to_thread(_download)
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Attachment content not found in storage.")

    async def delete(self, uri: str) -> None:
        def _delete():
            blob = self._bucket.blob(self._key_from_uri(uri))
            try:
                blob.delete()
            except Exception as e:
                logger.warning(f"Delete failed for {uri}: {e}")
        await asyncio.to_thread(_delete)

    async def move(self, old_uri: str, new_key: str) -> str:
        old_key = self._key_from_uri(old_uri)
        
        def _move():
            source_blob = self._bucket.blob(old_key)
            self._bucket.copy_blob(source_blob, self._bucket, new_key)
            source_blob.delete()
            
        await asyncio.to_thread(_move)
        logger.info("Moved object gs://%s/%s to %s", self.bucket_name, old_key, new_key)
        return f"{GCS_URI_PREFIX}{self.bucket_name}/{new_key}"
