"""Object storage backends for attachments.

Uploaded files are stored outside Firestore; only metadata is persisted in the
database. Concrete backends implement :class:`~app.storage.base.StorageBackend`.
"""

from __future__ import annotations

from app.core.config import settings
from app.storage.base import StorageBackend


def build_storage_backend() -> StorageBackend:
    """Factory selecting a backend from configuration.

    ``STORAGE_BACKEND=gcs`` (or ``STORAGE_BUCKET`` set) → GCS;
    otherwise a local-directory backend for development and tests.
    """
    from app.storage.gcs import GCSStorageBackend
    from app.storage.local import LocalStorageBackend

    backend = settings.storage_backend.strip().lower()
    if backend in ("", "auto"):
        backend = "gcs" if settings.storage_bucket else "local"
    if backend == "gcs":
        return GCSStorageBackend(bucket_name=settings.storage_bucket)
    if backend == "local":
        return LocalStorageBackend(root=settings.local_storage_dir)
    raise ValueError(f"Unknown storage backend: {backend!r}")
