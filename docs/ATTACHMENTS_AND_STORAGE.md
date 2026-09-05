# Attachments & Cloud Storage Architecture

This document describes the design and lifecycle of file uploads, object storage, and temporary signed URLs in the Cognito Chat API.

---

## 1. Core Architectural Principle

| Responsibility | Component | Role |
| :--- | :--- | :--- |
| **File Identity** | **Firestore** (`attachments`, `sessions/{id}/messages`) | Stores canonical, stable metadata (`bucket`, `object_name`, `filename`, `content_type`, `size`). **Never stores signed URLs.** |
| **Authorization & Temporary Access** | **Backend** (`AttachmentUrlService`, FastAPI) | Verifies user ownership and signs temporary GCS URLs on-the-fly when returning responses. |
| **Direct File Delivery** | **Google Cloud Storage (GCS)** | Handles high-bandwidth uploads and downloads directly with the client browser. No file proxying through FastAPI. |
| **Rendering** | **Frontend Client** (`cognito-chat`) | Consumes `attachment.url` directly. Contains **zero logic** for GCS authentication, signing, or token refresh. |

---

## 2. Architecture & Data Flow

### A. Direct Presigned Upload (Browser ──▶ GCS)

```text
Frontend (Browser)                FastAPI Backend                  Google Cloud Storage (GCS)
       │                                 │                                     │
       │── 1. POST /attachments/upload-url ──▶                               │
       │      (filename, mime, size)     │                                     │
       │                                 │── Creates AttachmentMetadata        │
       │                                 │   (bucket, object_name) in Firestore │
       │                                 │── Generates GCS signed PUT URL      │
       │◀── 2. PresignedUploadResponse ──│                                     │
       │      (upload_url, headers, att) │                                     │
       │                                                                       │
       │── 3. HTTP PUT (stream binary bytes directly to storage) ─────────────▶│
       │      (bypasses application backend completely)                        │
```

1. The client requests an upload ticket via `POST /agent/attachments/upload-url`.
2. The backend validates file constraints (size, allowed types, user quota), pre-creates the `AttachmentMetadata` document in Firestore with canonical storage references (`bucket`, `object_name`), and generates a v4 signed `PUT` URL.
3. The browser streams bytes directly to GCS via HTTP `PUT` with progress reporting, bypassing FastAPI.

---

### B. Conversation & Attachment Retrieval (Zero N+1 Pattern)

When fetching a conversation (`GET /agent/sessions/{session_id}` or `GET /agent/shared/{share_id}`):

```text
GET /agent/sessions/:id
       │
       ▼
1. Fetch session + messages from Firestore
       │
       ▼
2. Collect all attachment IDs across all message parts
       │
       ▼
3. Batch query Firestore once: AttachmentRepository.get_many(user_id, [ids])
   (Single retrieval path — enforces strict user ownership)
       │
       ▼
4. Concurrently sign fresh download URLs in memory: AttachmentUrlService.generate_attachment_url(...)
   (Default lifetime: 60 minutes with explicit url_expires_at)
       │
       ▼
5. Inject { url, urlExpiresAt, bucket, objectName } into message file parts
       │
       ▼
Return enriched response (1 HTTP request, 1 Firestore query batch)
```

- **Single Firestore query path**: All attachments for a conversation are resolved in a single batch query (`get_many`), eliminating N+1 database operations.
- **No FastAPI proxying**: The client fetches images and documents directly from GCS using the generated signed URL.
- **Graceful degradation**: If signing fails for an individual attachment, it logs a warning and sets `url: null` for that part without failing the conversation request.

---

## 3. Storage Identity Contract

### Firestore Model (`AttachmentMetadata`)

Canonical reference persisted in the `attachments` collection:

```json
{
  "id": "b1b017b2-c0e4-4fa9-bcf3-401777ea2f2c",
  "user_id": "7fa2a3c1-b0e1-4321-9988-112233445566",
  "session_id": "c9281a8b-1122-4433-8899-001122334455",
  "filename": "quarterly_results.pdf",
  "mime_type": "application/pdf",
  "size": 240182,
  "bucket": "cognito-production-attachments",
  "object_name": "attachments/7fa2a3c1/b1b017b2/quarterly_results.pdf",
  "storage_uri": "gs://cognito-production-attachments/attachments/7fa2a3c1/b1b017b2/quarterly_results.pdf",
  "type": "pdf",
  "is_temporary": false,
  "uploaded_at": "2026-09-05T11:00:00.000Z"
}
```

> **Crucial Rule**: The database record contains **no signed URL and no expiration timestamp**.

### Wire Schema (`AttachmentSchema` / Message `file` Part)

Returned dynamically to the frontend client:

```json
{
  "id": "b1b017b2-c0e4-4fa9-bcf3-401777ea2f2c",
  "filename": "quarterly_results.pdf",
  "contentType": "application/pdf",
  "mimeType": "application/pdf",
  "size": 240182,
  "bucket": "cognito-production-attachments",
  "objectName": "attachments/7fa2a3c1/b1b017b2/quarterly_results.pdf",
  "url": "https://storage.googleapis.com/cognito-production-attachments/attachments/...?X-Goog-Signature=...",
  "urlExpiresAt": "2026-09-05T12:00:00.000Z",
  "uploadedAt": "2026-09-05T11:00:00.000Z"
}
```

---

## 4. Backend Abstraction: `AttachmentUrlService`

All signed URL logic is consolidated in [`app/services/attachment_url.py`](file:///Users/bobby/Documents/Workspace/project-cognito/cognito-chat-api/app/services/attachment_url.py):

```python
class AttachmentUrlService:
    def __init__(self, storage: StorageBackend): ...

    async def generate_attachment_url(
        self, attachment: AttachmentMetadata | str, expires_in: int = 3600
    ) -> tuple[str | None, datetime | None]: ...

    async def enrich_attachment(self, metadata: AttachmentMetadata, expires_in: int = 3600) -> AttachmentSchema: ...

    async def enrich_attachments(
        self, metadatas: Sequence[AttachmentMetadata], expires_in: int = 3600
    ) -> list[AttachmentSchema]: ...

    async def enrich_message_attachments(
        self, messages: Sequence[Any], user_id: UUID | str, att_repo: Any, expires_in: int = 3600
    ) -> None: ...
```

Injected into FastAPI route handlers via the standard dependency:
```python
url_service: AttachmentUrlService = Depends(get_attachment_url_service)
```

---

## 5. Security & Authorization

1. **User Ownership Validation**:
   - `enrich_message_attachments` queries `att_repo.get_many(user_id, list(attachment_ids))`, ensuring that only attachments owned by the calling `user_id` are fetched and signed.
   - User A cannot obtain a signed URL for User B's attachment simply by passing User B's attachment ID in their session messages.
2. **Public Shared Chats**:
   - Public shared chats (`/agent/shared/{share_id}`) check share revocation and pre-verify that referenced attachments belong to the author's snapshot before generating download URLs. Revoking a share immediately cuts off signed URL generation.
3. **Short-Lived Credentials**:
   - Default signed URL expiration is set to 60 minutes (`3600s`), avoiding permanent public exposure of private bucket objects.

---

## 6. Database Migrations (`scripts/migrations/attachments/`)

A dedicated idempotent migration is provided to migrate existing environments:

### Running the Migration

```bash
# Run migration specifically for attachments
make migrate attachments
# OR
python scripts/migrate.py attachments

# List all available migrations
python scripts/migrate.py --list
```

### What Step `001_backfill_storage_identity_and_clean_urls` Does:
1. **Backfills Canonical Identity**: Populates `bucket` and `object_name` for legacy `attachments` records from their `storage_uri`.
2. **Purges Transient URLs**: Deletes any stale `url` or `url_expires_at` fields accidentally stored in Firestore `attachments` or `messages` documents.
3. **Cleans Shared Chat Snapshots**: Strips temporary URLs from frozen snapshot messages.
4. **Invalidates Redis Caches**: Clears `attachments:*` and `sessions:*` prefixes so fresh schemas are served immediately.
