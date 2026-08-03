"""Tests for the local storage backend."""

import pytest

from app.storage.local import LocalStorageBackend


@pytest.mark.asyncio
async def test_upload_read_delete_roundtrip(tmp_path):
    backend = LocalStorageBackend(root=str(tmp_path))

    uri = await backend.upload_bytes("attachments/u1/img.png", b"image-bytes", "image/png")
    assert uri == "local://attachments/u1/img.png"

    assert await backend.read_bytes(uri) == b"image-bytes"

    await backend.delete(uri)
    with pytest.raises(FileNotFoundError):
        await backend.read_bytes(uri)


@pytest.mark.asyncio
async def test_path_traversal_is_rejected(tmp_path):
    backend = LocalStorageBackend(root=str(tmp_path))
    with pytest.raises(ValueError):
        await backend.read_bytes("local://../../etc/passwd")


@pytest.mark.asyncio
async def test_delete_missing_is_noop(tmp_path):
    backend = LocalStorageBackend(root=str(tmp_path))
    await backend.delete("local://does/not/exist")
