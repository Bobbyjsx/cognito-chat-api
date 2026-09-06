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

    def _get_signing_credentials_and_email(self):
        """Returns (credentials, service_account_email, access_token) for URL signing."""
        creds = getattr(self._client, "_credentials", None)
        if creds is None:
            return None, None, None

        try:
            from google.auth.credentials import Signing

            if isinstance(creds, Signing):
                return creds, None, None
        except Exception as e:
            logger.debug("Credentials signing check skipped: %s", e)

        # Environment without local private key (Cloud Run / GCE)
        try:
            from google.auth.transport.requests import Request

            # Ensure scopes include cloud-platform for IAM signBlob
            if hasattr(creds, "with_scopes"):
                try:
                    creds = creds.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
                except Exception as e:
                    logger.debug("Failed adding cloud-platform scope: %s", e)

            if hasattr(creds, "valid") and not creds.valid and hasattr(creds, "refresh"):
                creds.refresh(Request())
        except Exception as e:
            logger.debug("Failed refreshing credentials for signing: %s", e)

        sa_email = getattr(creds, "service_account_email", None)
        if not sa_email or sa_email == "default":
            try:
                from app.core.config import settings

                sa_email = settings.gcs_service_account_email or None
            except Exception:
                sa_email = None

        if not sa_email:
            try:
                import urllib.request

                req = urllib.request.Request(
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
                    headers={"Metadata-Flavor": "Google"},
                )
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        email = resp.read().decode("utf-8").strip()
                        if email:
                            sa_email = email
            except Exception as e:
                logger.debug("Metadata server email lookup skipped: %s", e)

        token = getattr(creds, "token", None)
        return creds, sa_email, token

    @staticmethod
    def _build_response_disposition(filename: str | None) -> str:
        if not filename:
            return "attachment"
        import urllib.parse

        ascii_filename = filename.encode("ascii", "ignore").decode("ascii").replace('"', "")
        encoded_filename = urllib.parse.quote(filename, safe="")
        if ascii_filename:
            return f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"
        return f"attachment; filename*=UTF-8''{encoded_filename}"

    async def generate_upload_url(
        self, key: str, content_type: str, expires_in: int = 1800
    ) -> tuple[str, dict[str, str]]:
        from datetime import timedelta

        creds, sa_email, access_token = self._get_signing_credentials_and_email()

        def _sign():
            blob = self._bucket.blob(key)
            kwargs = {
                "version": "v4",
                "expiration": timedelta(seconds=expires_in),
                "method": "PUT",
                "content_type": content_type,
            }
            if access_token and sa_email:
                kwargs["service_account_email"] = sa_email
                kwargs["access_token"] = access_token
            elif creds:
                kwargs["credentials"] = creds
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
        creds, sa_email, access_token = self._get_signing_credentials_and_email()

        def _sign():
            blob = self._bucket.blob(key)
            kwargs = {
                "version": "v4",
                "expiration": timedelta(seconds=expires_in),
                "method": "GET",
            }
            if disposition == "attachment":
                kwargs["response_disposition"] = self._build_response_disposition(filename)
            if access_token and sa_email:
                kwargs["service_account_email"] = sa_email
                kwargs["access_token"] = access_token
            elif creds:
                kwargs["credentials"] = creds
            return blob.generate_signed_url(**kwargs)

        return await asyncio.to_thread(_sign)
