import json
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import AlreadyExists
from google.cloud.firestore_admin_v1 import types

from scripts.deploy_indexes import (
    build_index_proto,
    deploy_indexes,
    format_index_desc,
    load_indexes_file,
    normalize_existing_index,
    normalize_target_index,
)


def test_normalize_target_index():
    target = {
        "collectionGroup": "sessions",
        "queryScope": "COLLECTION",
        "fields": [
            {"fieldPath": "user_id", "order": "ASCENDING"},
            {"fieldPath": "updated_at", "order": "DESCENDING"},
        ],
    }
    sig = normalize_target_index(target)
    assert sig == (
        "sessions",
        "COLLECTION",
        (("user_id", "ASCENDING", None), ("updated_at", "DESCENDING", None)),
    )


def test_normalize_existing_index():
    idx_proto = types.Index(
        name="projects/test-p/databases/test-d/collectionGroups/sessions/indexes/12345",
        query_scope=types.Index.QueryScope.COLLECTION,
        fields=[
            types.Index.IndexField(
                field_path="user_id",
                order=types.Index.IndexField.Order.ASCENDING,
            ),
            types.Index.IndexField(
                field_path="updated_at",
                order=types.Index.IndexField.Order.DESCENDING,
            ),
        ],
    )
    sig = normalize_existing_index(idx_proto)
    assert sig == (
        "sessions",
        "COLLECTION",
        (("user_id", "ASCENDING", None), ("updated_at", "DESCENDING", None)),
    )


def test_signatures_match_between_target_and_proto():
    target = {
        "collectionGroup": "attachments",
        "queryScope": "COLLECTION",
        "fields": [
            {"fieldPath": "user_id", "order": "ASCENDING"},
            {"fieldPath": "is_temporary", "order": "ASCENDING"},
            {"fieldPath": "uploaded_at", "order": "DESCENDING"},
        ],
    }
    idx_proto = types.Index(
        name="projects/p/databases/d/collectionGroups/attachments/indexes/abc",
        query_scope=types.Index.QueryScope.COLLECTION,
        fields=[
            types.Index.IndexField(
                field_path="user_id",
                order=types.Index.IndexField.Order.ASCENDING,
            ),
            types.Index.IndexField(
                field_path="is_temporary",
                order=types.Index.IndexField.Order.ASCENDING,
            ),
            types.Index.IndexField(
                field_path="uploaded_at",
                order=types.Index.IndexField.Order.DESCENDING,
            ),
        ],
    )
    assert normalize_target_index(target) == normalize_existing_index(idx_proto)


def test_build_index_proto():
    target = {
        "collectionGroup": "sessions",
        "queryScope": "COLLECTION",
        "fields": [
            {"fieldPath": "user_id", "order": "ASCENDING"},
            {"fieldPath": "updated_at", "order": "DESCENDING"},
        ],
    }
    proto = build_index_proto(target)
    assert proto.query_scope == types.Index.QueryScope.COLLECTION
    assert len(proto.fields) == 2
    assert proto.fields[0].field_path == "user_id"
    assert proto.fields[0].order == types.Index.IndexField.Order.ASCENDING
    assert proto.fields[1].field_path == "updated_at"
    assert proto.fields[1].order == types.Index.IndexField.Order.DESCENDING


def test_load_indexes_file(tmp_path):
    valid_file = tmp_path / "indexes.json"
    valid_file.write_text(json.dumps({"indexes": [{"collectionGroup": "test", "fields": []}]}))
    loaded = load_indexes_file(valid_file)
    assert len(loaded) == 1

    missing_file = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_indexes_file(missing_file)

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text(json.dumps({"indexes": "not-a-list"}))
    with pytest.raises(TypeError):
        load_indexes_file(invalid_file)


def test_format_index_desc():
    desc = format_index_desc("sessions", [{"fieldPath": "user_id", "order": "ASCENDING"}])
    assert desc == "sessions [user_id (ASCENDING)]"


def test_deploy_indexes_emulator_skip(monkeypatch, tmp_path):
    monkeypatch.setenv("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8080")
    dummy_file = tmp_path / "indexes.json"
    dummy_file.write_text(json.dumps({"indexes": []}))

    with patch("scripts.deploy_indexes.create_admin_client") as mock_client:
        success = deploy_indexes(dummy_file, "test-p", "test-d")
        assert success is True
        mock_client.assert_not_called()


def test_deploy_indexes_skips_existing(monkeypatch, tmp_path):
    monkeypatch.delenv("FIRESTORE_EMULATOR_HOST", raising=False)
    index_file = tmp_path / "indexes.json"
    index_file.write_text(
        json.dumps(
            {
                "indexes": [
                    {
                        "collectionGroup": "sessions",
                        "queryScope": "COLLECTION",
                        "fields": [
                            {"fieldPath": "user_id", "order": "ASCENDING"},
                        ],
                    }
                ]
            }
        )
    )

    existing_proto = types.Index(
        name="projects/p/databases/d/collectionGroups/sessions/indexes/1",
        query_scope=types.Index.QueryScope.COLLECTION,
        fields=[
            types.Index.IndexField(
                field_path="user_id",
                order=types.Index.IndexField.Order.ASCENDING,
            )
        ],
    )

    mock_client = MagicMock()
    mock_client.collection_group_path.return_value = "projects/p/databases/d/collectionGroups/-"
    mock_client.list_indexes.return_value = [existing_proto]

    with patch("scripts.deploy_indexes.create_admin_client", return_value=mock_client):
        success = deploy_indexes(index_file, "p", "d")
        assert success is True
        mock_client.create_index.assert_not_called()


def test_deploy_indexes_creates_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("FIRESTORE_EMULATOR_HOST", raising=False)
    index_file = tmp_path / "indexes.json"
    index_file.write_text(
        json.dumps(
            {
                "indexes": [
                    {
                        "collectionGroup": "sessions",
                        "queryScope": "COLLECTION",
                        "fields": [
                            {"fieldPath": "user_id", "order": "ASCENDING"},
                        ],
                    }
                ]
            }
        )
    )

    mock_client = MagicMock()
    mock_client.collection_group_path.return_value = "projects/p/databases/d/collectionGroups/-"
    mock_client.list_indexes.return_value = []
    mock_op = MagicMock()
    mock_op.operation.name = "projects/p/databases/d/operations/op1"
    mock_client.create_index.return_value = mock_op

    with patch("scripts.deploy_indexes.create_admin_client", return_value=mock_client):
        success = deploy_indexes(index_file, "p", "d")
        assert success is True
        mock_client.create_index.assert_called_once()


def test_deploy_indexes_handles_already_exists_gracefully(monkeypatch, tmp_path):
    monkeypatch.delenv("FIRESTORE_EMULATOR_HOST", raising=False)
    index_file = tmp_path / "indexes.json"
    index_file.write_text(
        json.dumps(
            {
                "indexes": [
                    {
                        "collectionGroup": "sessions",
                        "queryScope": "COLLECTION",
                        "fields": [
                            {"fieldPath": "user_id", "order": "ASCENDING"},
                        ],
                    }
                ]
            }
        )
    )

    mock_client = MagicMock()
    mock_client.collection_group_path.return_value = "projects/p/databases/d/collectionGroups/-"
    mock_client.list_indexes.return_value = []
    mock_client.create_index.side_effect = AlreadyExists("Index already exists")

    with patch("scripts.deploy_indexes.create_admin_client", return_value=mock_client):
        success = deploy_indexes(index_file, "p", "d")
        assert success is True
        mock_client.create_index.assert_called_once()


def test_deploy_indexes_dry_run(monkeypatch, tmp_path):
    monkeypatch.delenv("FIRESTORE_EMULATOR_HOST", raising=False)
    index_file = tmp_path / "indexes.json"
    index_file.write_text(
        json.dumps(
            {
                "indexes": [
                    {
                        "collectionGroup": "sessions",
                        "queryScope": "COLLECTION",
                        "fields": [
                            {"fieldPath": "user_id", "order": "ASCENDING"},
                        ],
                    }
                ]
            }
        )
    )

    mock_client = MagicMock()
    mock_client.collection_group_path.return_value = "projects/p/databases/d/collectionGroups/-"
    mock_client.list_indexes.return_value = []

    with patch("scripts.deploy_indexes.create_admin_client", return_value=mock_client):
        success = deploy_indexes(index_file, "p", "d", dry_run=True)
        assert success is True
        mock_client.create_index.assert_not_called()
