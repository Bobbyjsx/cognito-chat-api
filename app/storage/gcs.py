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
        self.bucket_name = bucket_name
        self._client = client or self._create_client()
        self._bucket = self._client.bucket(self.bucket_name)

    @staticmethod
    def _create_client():
        import os

        from google.cloud.storage import Client

        from app.core.config import settings

        # 1. Prefer explicit Firebase / GCP service account file from config
        cred_path = getattr(settings, "firebase_credentials_path", None)
        if cred_path and os.path.exists(cred_path):
            try:
                return Client.from_service_account_json(cred_path)
            except Exception as e:
                logger.warning("Failed to initialize GCS client from firebase_credentials_path: %s", e)

        # 2. Check GOOGLE_APPLICATION_CREDENTIALS environment variable
        google_app_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if google_app_creds and os.path.exists(google_app_creds):
            try:
                return Client.from_service_account_json(google_app_creds)
            except Exception as e:
                logger.warning("Failed to initialize GCS client from GOOGLE_APPLICATION_CREDENTIALS: %s", e)

        # 3. Fallback to default client
        return Client()

    def _get_signing_extra(self) -> dict[str, str]:
        """Provide service_account_email and access_token when credentials lack local private keys."""
        try:
            from google.auth.credentials import Signing

            creds = getattr(self._client, "_credentials", None) or getattr(self._client, "credentials", None)
            if creds is None or isinstance(creds, Signing):
                return {}

            extra: dict[str, str] = {}
            sa_email = getattr(creds, "service_account_email", None)
            if not sa_email:
                from app.core.config import settings

                sa_email = getattr(settings, "cloud_tasks_service_account_email", None)
            if sa_email:
                extra["service_account_email"] = sa_email

            if not getattr(creds, "valid", False):
                from google.auth.transport.requests import Request as AuthRequest

                creds.refresh(AuthRequest())
            token = getattr(creds, "token", None)
            if token:
                extra["access_token"] = token
            return extra
        except Exception as e:
            logger.debug("Could not resolve extra IAM signing parameters: %s", e)
            return {}

    @staticmethod
    def _key_from_uri(uri: str) -> str:
        if uri.startswith(GCS_URI_PREFIX):
            return uri[len(GCS_URI_PREFIX) :].split("/", 1)[1]
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

    async def generate_upload_url(
        self, key: str, content_type: str, expires_in: int = 1800
    ) -> tuple[str, dict[str, str]]:
        from datetime import timedelta

        def _sign():
            blob = self._bucket.blob(key)
            kwargs = {
                "version": "v4",
                "expiration": timedelta(seconds=expires_in),
                "method": "PUT",
                "content_type": content_type,
            }
            kwargs.update(self._get_signing_extra())
            url = blob.generate_signed_url(**kwargs)
            return url, {"Content-Type": content_type}

        return await asyncio.to_thread(_sign)

    async def generate_download_url(
        self,
        uri: str,
        expires_in: int = 7200,
        filename: str | None = None,
        disposition: str = "inline",
    ) -> str:
        from datetime import timedelta

        key = self._key_from_uri(uri)

        def _sign():
            blob = self._bucket.blob(key)
            kwargs = {
                "version": "v4",
                "expiration": timedelta(seconds=expires_in),
                "method": "GET",
            }
            if disposition == "attachment":
                kwargs["response_disposition"] = f'attachment; filename="{filename}"' if filename else "attachment"
            kwargs.update(self._get_signing_extra())
            return blob.generate_signed_url(**kwargs)

        return await asyncio.to_thread(_sign)
