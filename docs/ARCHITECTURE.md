# Cognito Chat API — Architecture

This document describes the architecture after the Gemini tools & attachments
evolution. `API_OVERVIEW.md` remains a historical overview of the original
system; where the two disagree, this document wins.

## System overview

```
Client ──▶ HTTP API (FastAPI)
             ├── /auth            auth + quota management
             ├── /agent/chat      chat (SSE) + non-streaming chat
             ├── /agent/sessions  session persistence
             ├── /agent/attachments  attachment upload / lookup / delete
             └── /config          admin config, /agent/stt transcription

App services ──▶ app.services
    AgentService (composes everything) ──▶ ToolExecutor ──▶ ToolRegistry ──▶ Tools
        │                                  │
        ├──▶ GeminiProvider (app.providers)  └──▶ Gemini API (google-genai SDK)
        ├──▶ AttachmentService ──▶ StorageBackend (GCS or local)
        ├──▶ ContextManager (history trimming)
        └──▶ ConfigRepository / UserRepository / ChatRepository ──▶ Firestore
```

The Gemini SDK is confined to `app/providers/gemini.py`. Everything above it
talks to the provider through `BaseProvider` and pure dataclasses; the rest of
the app no longer imports `google.genai` types.

## Provider responsibilities (`app/providers/`)

`BaseProvider` defines the contract used by services:

- `generate(model, contents, config=None)` → `GenerationResult` (text,
  total tokens, `tool_calls`)
- `stream(model, contents, config=None)` → async iterator of `GenerationEvent`
  (`text`, `reasoning`, `tool_call`, `tool_result`, `usage`)
- `parts_for_attachment(metadata, data)` → SDK-agnostic content parts for a
  stored attachment
- `transcribe_audio(model, data, mime_type, prompt)` → (transcript, tokens)

`GeminiProvider` maps these to the google-genai SDK:

- Contents/tools/config conversion (`_to_sdk_contents`, `_to_sdk_config`,
  `build_tools` for `code_execution`, `google_search`, and function tools).
- Stream parsing: text/thought chunks, `function_call` parts, executable code
  parts, and code-execution result parts are translated into the internal
  event vocabulary. Google Search results arrive as grounding metadata, which
  the provider synthesizes into `tool_call`/`tool_result` events so the
  client-visible SSE contract is identical to function tools.
- Attachments: images and PDFs ≤ 20 MB are sent inline; audio ≤ 9 MB inline,
  otherwise (and all video) uploaded once to the Gemini Files API via
  `files.upload`, with the resulting URI cached in
  `AttachmentMetadata.gemini_file_uri` in Firestore.
- Errors are classified via `classify_provider_error` into
  `ProviderError` / `ProviderModelNotFoundError` / `ProviderGenerationError`
  with stable error codes, so the rest of the app never inspects SDK
  exceptions.

## Tool framework (`app/tools/`)

- `BaseTool` defines `kind` (`"function"` — a Gemini function-declaration
  tool — or `"server"` — a provider-side tool such as code execution /
  Google Search), `name`, `description`, `schema`, and `execute()`.
- `ToolRegistry` registers tools and `register_defaults()` adds
  `CodeExecutionTool` and `GoogleSearchTool`. `to_provider_configs()` emits
  only the tools enabled for a request.
- `ToolExecutor` runs the generation loop: it calls the provider, executes
  any returned function calls against the registry, appends the results as
  function-response parts, and re-invokes the provider, up to
  `MAX_TOOL_ITERATIONS = 4`. Tool failures are captured into the tool result
  so a broken tool never kills the conversation. `generate` returns the final
  non-tool response; `stream` yields the SSE events.
- `AgentService` contains no hardcoded tool list — tools are resolved entirely
  through the registry.

## Attachments (`app/attachments` pipeline)

Storage layout:

- **Firestore** stores only `AttachmentMetadata` (id, user_id, session_id,
  filename, mime_type, size, type, storage_uri, gemini_file_uri, uploaded_at)
  in the `attachments` collection. Bytes never live in Firestore.
- **Object storage** is behind `StorageBackend` (`upload_bytes` / `read_bytes`
  / `delete`). Two implementations:
  - `LocalStorageBackend` (`local://` URIs, files under `LOCAL_STORAGE_DIR`)
    for development/tests.
  - `GCSStorageBackend` (`gs://` URIs) for production; the import of
    `google.cloud.storage` is deferred so the library is only required when
    GCS is actually configured.
  - Selection: `STORAGE_BACKEND` env var, or automatic — GCS when
    `STORAGE_BUCKET` is set, otherwise local.

Upload flow (`POST /agent/attachments`):

1. Validate against runtime config: `enable_attachments` (403), per-file size
   ≤ `attachment_max_size` (413), MIME type in `attachment_allowed_types` (400).
2. MIME type is detected from magic bytes (`app/utils/mime.py`) and the file
   classified (`image`, `pdf`, `document`, `audio`, `video`, `spreadsheet`,
   `json`, `text`).
3. Bytes are uploaded to the storage backend; metadata is persisted in
   Firestore; the wire schema (no `gemini_file_uri`) is returned with 201.

Chat flow (`POST /agent/chat` with `attachments: [uuid, ...]`):

1. Attachment IDs are validated, ownership is checked, and unbound
   attachments are bound to the message's session (first use wins).
2. For each attachment, `AttachmentService.prepare_parts` either extracts text
   (txt/markdown/CSV/JSON/plain text/DOCX/XLSX) into a text part, or delegates
   to `provider.parts_for_attachment` for image/PDF/audio/video.
3. The message is persisted with its `attachment_ids`, and history built from
   those messages re-resolves historical attachments by ID, so past
   attachments survive a session restart. Broken historical attachments are
   tolerated (skipped with a log) rather than failing the request.

## Context management (`app/services/context.py`)

`ContextManager` estimates tokens at ~4 characters per token (plus a fixed
cost per attachment id) and, when enabled (`context_trim_enabled`), trims the
history sent to the provider to `context_max_tokens`, always keeping the
`context_keep_recent` most recent messages. Trimming applies to the provider
payload only — full history remains in Firestore.

## Configuration

- `AppConfigDB` (runtime, editable via `/config`) gained:
  `enable_attachments`, `attachment_max_size`, `attachment_max_count`,
  `attachment_allowed_types`, `context_trim_enabled`, `context_max_tokens`,
  `context_keep_recent`. All have defaults, so existing Firestore config
  documents work unchanged.
- `Settings` (env) gained `STORAGE_BACKEND`, `STORAGE_BUCKET`,
  `LOCAL_STORAGE_DIR`. See `.env.example`.
- `requirements.txt` added `google-cloud-storage` (only needed at runtime when
  the GCS backend is used).

## Client-visible API

Unchanged endpoints remain byte-compatible. New additions:

- `POST /agent/attachments` (multipart: `file`, optional `session_id`) → 201
  attachment schema; 400 unsupported type, 403 disabled, 413 too large.
- `GET /agent/attachments?session_id=<uuid>` — list (optionally by session).
- `GET /agent/attachments/{id}` — single attachment.
- `DELETE /agent/attachments/{id}` — delete metadata + stored bytes.
- `POST /agent/chat` and `/agent/chat/stream` accept `attachments:
  [uuid, ...]`; `MessageSchema` gains `attachment_ids`. Responses/SSE events
  are unchanged (`text`, `reasoning`, `tool_call`, `tool_result`, `usage`).
- `/config` response now includes `allowed_text_models` (previously the field
  was not serialized — bug fix).

## Migration notes

- No Firestore data migration required: all new message/config fields have
  defaults and the `attachments` collection is new. Composite index
  `attachments: user_id ASC, uploaded_at DESC` is defined in
  `firestore.indexes.json` and must be deployed (`firebase deploy --only
  firestore:indexes`).
- Deploy GCS: set `STORAGE_BACKEND=gcs` and `STORAGE_BUCKET`; ensure the
  service account can read/write that bucket.
- If previously using the `/agent/stt` endpoint with uploads, consider
  switching to chat attachments (audio) — the STT endpoint is retained and
  now delegates to the provider's `transcribe_audio`.
- Deleting a user's attachments manually requires deleting both the Firestore
  document and the storage object; `DELETE /agent/attachments/{id}` does both.
