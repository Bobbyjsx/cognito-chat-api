# Cognito Chat API — Architecture

This document describes the architecture of the Cognito Chat API, including the Gemini tools & attachments pipeline, and the transparent server-side authentication & token lifecycle architecture.

## System overview

```
Client (Web/Mobile) ──▶ Direct HTTP API (FastAPI)
                          ├── /auth/signup, /auth/login, /auth/refresh, /auth/me
                          ├── /agent/chat       chat (SSE) + non-streaming chat
                          ├── /agent/sessions   session persistence & management
                          ├── /agent/attachments attachment upload / lookup / delete / library
                          └── /config           app config, /agent/stt transcription

App services ──▶ app.services
    AgentService (composes everything) ──▶ ToolExecutor ──▶ ToolRegistry ──▶ Tools
        │                                  │
        ├──▶ GeminiProvider (app.providers)  └──▶ Gemini API (google-genai SDK)
        ├──▶ AttachmentService ──▶ StorageBackend (GCS or local)
        ├──▶ ContextManager (history trimming)
        └──▶ ConfigRepository / UserRepository / ChatRepository ──▶ Firestore
```

The Gemini SDK is confined to `app/providers/gemini.py`. Everything above it talks to the provider through `BaseProvider` and pure dataclasses; the rest of the app no longer imports `google.genai` types.

---

## Authentication & Server-Side Token Management (`app/core/token_manager.py`, `app/api/dependencies.py`)

Token lifecycle management is completely transparent to the client. The client does not need to catch `401 Unauthorized` responses in browser JavaScript, manage refresh loops, or re-issue failed requests.

### Request/Response Header Contract

| Direction | Header | Description |
| :--- | :--- | :--- |
| **Client ──▶ Server** | `Authorization: Bearer <access_token>` | Current access token |
| **Client ──▶ Server** | `x-refresh-token: <refresh_token>` | Current refresh token (optional, enables auto-refresh) |
| **Server ──▶ Client** | `x-new-access-token: <new_access_token>` | Emitted on 2xx responses when access token was refreshed |
| **Server ──▶ Client** | `x-new-refresh-token: <new_refresh_token>` | Emitted on 2xx responses when refresh token was rotated |

### Flow & Dependencies

1. **Direct Request**: The client sends direct API requests with `Authorization: Bearer <access_token>` and `x-refresh-token: <refresh_token>`.
2. **`get_current_user` Dependency**:
   - Validates access token signature and expiration against JWKS (Identity Service) or local secret key.
   - If token is **valid and unexpired** (outside the 60s buffer): returns authenticated user.
   - If token is **near expiry** (exp <= 60s) or **expired** (`ExpiredSignatureError`):
     - If `x-refresh-token` is present: executes single-flight token refresh via `ServerTokenRefreshManager`.
     - Attaches `x-new-access-token` and `x-new-refresh-token` to `response.headers`.
     - Continues execution and returns `200 OK` on the first attempt without failing or requiring a client retry.
     - If `x-refresh-token` is missing or invalid: raises `401 Unauthorized`.
3. **Single-Flight Concurrency Deduplication**:
   - Managed by `ServerTokenRefreshManager` using per-token `asyncio.Lock` and a 5-second burst cache.
   - When multiple concurrent requests arrive with an expired token, exactly **one** refresh request is executed against the identity/auth service; all awaiting requests receive the updated credentials concurrently.
4. **CORS Exposure**:
   - `CORSMiddleware` exposes `X-New-Access-Token` and `X-New-Refresh-Token` headers so client browser interceptors can read them and update local session state in the background.

---

## Provider responsibilities (`app/providers/`)

`BaseProvider` defines the contract used by services:

- `generate(model, contents, config=None)` → `GenerationResult` (text, total tokens, `tool_calls`)
- `stream(model, contents, config=None)` → async iterator of `GenerationEvent` (`text`, `reasoning`, `tool_call`, `tool_result`, `usage`)
- `parts_for_attachment(metadata, data)` → SDK-agnostic content parts for a stored attachment
- `transcribe_audio(model, data, mime_type, prompt)` → (transcript, tokens)

`GeminiProvider` maps these to the google-genai SDK:

- Contents/tools/config conversion (`_to_sdk_contents`, `_to_sdk_config`, `build_tools` for `code_execution`, `google_search`, and function tools).
- Stream parsing: text/thought chunks, `function_call` parts, executable code parts, and code-execution result parts are translated into the internal event vocabulary. Google Search results arrive as grounding metadata, which the provider synthesizes into `tool_call`/`tool_result` events so the client-visible SSE contract is identical to function tools.
- Attachments: images and PDFs ≤ 20 MB are sent inline; audio ≤ 9 MB inline, otherwise (and all video) uploaded once to the Gemini Files API via `files.upload`, with the resulting URI cached in `AttachmentMetadata.gemini_file_uri` in Firestore.
- Errors are classified via `classify_provider_error` into `ProviderError` / `ProviderModelNotFoundError` / `ProviderGenerationError` with stable error codes, so the rest of the app never inspects SDK exceptions.

---

## Tool framework (`app/tools/`)

- `BaseTool` defines `kind` (`"function"` — a Gemini function-declaration tool — or `"server"` — a provider-side tool such as code execution / Google Search), `name`, `description`, `schema`, and `execute()`.
- `ToolRegistry` registers tools and `register_defaults()` adds `CodeExecutionTool` and `GoogleSearchTool`. `to_provider_configs()` emits only the tools enabled for a request.
- `ToolExecutor` runs the generation loop: it calls the provider, executes any returned function calls against the registry, appends the results as function-response parts, and re-invokes the provider, up to `MAX_TOOL_ITERATIONS = 4`. Tool failures are captured into the tool result so a broken tool never kills the conversation. `generate` returns the final non-tool response; `stream` yields the SSE events.
- `AgentService` contains no hardcoded tool list — tools are resolved entirely through the registry.

---

## Attachments (`app/attachments` pipeline)

Storage layout:

- **Firestore** stores only `AttachmentMetadata` (id, user_id, session_id, filename, mime_type, size, type, storage_uri, gemini_file_uri, uploaded_at) in the `attachments` collection. Bytes never live in Firestore.
- **Object storage** is behind `StorageBackend` (`upload_bytes` / `read_bytes` / `delete`). Two implementations:
  - `LocalStorageBackend` (`local://` URIs, files under `LOCAL_STORAGE_DIR`) for development/tests.
  - `GCSStorageBackend` (`gs://` URIs) for production; the import of `google.cloud.storage` is deferred so the library is only required when GCS is actually configured.
  - Selection: `STORAGE_BACKEND` env var, or automatic — GCS when `STORAGE_BUCKET` is set, otherwise local.

Upload flow (`POST /agent/attachments`):

1. Validate against runtime config: `enable_attachments` (403), per-file size ≤ `attachment_max_size` (413), MIME type in `attachment_allowed_types` (400).
2. MIME type is detected from magic bytes (`app/utils/mime.py`) and the file classified (`image`, `pdf`, `document`, `audio`, `video`, `spreadsheet`, `json`, `text`).
3. Bytes are uploaded to the storage backend; metadata is persisted in Firestore; the wire schema (no `gemini_file_uri`) is returned with 201.

Chat flow (`POST /agent/chat` with `attachments: [uuid, ...]`):

1. Attachment IDs are validated, ownership is checked, and unbound attachments are bound to the message's session (first use wins).
2. For each attachment, `AttachmentService.prepare_parts` either extracts text (txt/markdown/CSV/JSON/plain text/DOCX/XLSX) into a text part, or delegates to `provider.parts_for_attachment` for image/PDF/audio/video.
3. The message is persisted with its `attachment_ids`, and history built from those messages re-resolves historical attachments by ID, so past attachments survive a session restart. Broken historical attachments are tolerated (skipped with a log) rather than failing the request.

---

## Context management (`app/services/context.py`)

`ContextManager` estimates tokens at ~4 characters per token (plus a fixed cost per attachment id) and, when enabled (`context_trim_enabled`), trims the history sent to the provider to `context_max_tokens`, always keeping the `context_keep_recent` most recent messages. Trimming applies to the provider payload only — full history remains in Firestore.

---

## Configuration

- `AppConfigDB` (runtime, editable via `/config`) gained: `enable_attachments`, `attachment_max_size`, `attachment_max_count`, `attachment_allowed_types`, `context_trim_enabled`, `context_max_tokens`, `context_keep_recent`. All have defaults, so existing Firestore config documents work unchanged.
- `Settings` (env) gained `STORAGE_BACKEND`, `STORAGE_BUCKET`, `LOCAL_STORAGE_DIR`. See `.env.example`.
- `requirements.txt` added `google-cloud-storage` (only needed at runtime when the GCS backend is used).

---

## Client-visible API

- `POST /agent/attachments` (multipart: `file`, optional `session_id`) → 201 attachment schema; 400 unsupported type, 403 disabled, 413 too large.
- `GET /agent/attachments?session_id=<uuid>` — list (optionally by session).
- `GET /agent/attachments/{id}` — single attachment.
- `DELETE /agent/attachments/{id}` — delete metadata + stored bytes.
- `POST /agent/chat` and `/agent/chat/stream` accept `attachments: [uuid, ...]`; `MessageSchema` gains `attachment_ids`. Responses/SSE events are unchanged (`text`, `reasoning`, `tool_call`, `tool_result`, `usage`).
- `/config` response now includes `allowed_text_models`.
- Response headers on protected endpoints: `X-New-Access-Token` and `X-New-Refresh-Token` when rotated.

---

## Database & Feature Migration Architecture (`scripts/migrations/`)

Database and configuration migrations are organized in **feature-scoped modules** under `scripts/migrations/`.

### Directory Layout

```
scripts/
├── migrate.py                      # Central CLI & dependency dispatcher
├── migrations/
│   ├── core_config/                # Feature: core-config / initial-setup
│   │   ├── runner.py               # Step registry & execution runner
│   │   └── 001_seed_app_config.py
│   ├── user_auth/                  # Feature: user-auth / users
│   │   ├── runner.py
│   │   ├── 001_migrate_users_schema.py
│   │   └── 002_migrate_quota_limits.py
│   ├── speech_to_text/             # Feature: speech-to-text / stt
│   │   ├── runner.py
│   │   └── 001_migrate_stt_config.py
│   └── smart_model_routing/        # Feature: smart-model-routing
│       ├── runner.py
│       ├── 001_migrate_models_list_structure.py
│       ├── 002_migrate_model_descriptions.py
│       └── 003_migrate_unified_effort_modes.py
```

### Key Principles

1. **Feature Scoping**: Every migration belongs to a specific feature directory.
2. **Chronological Ordering**: Each feature contains numbered steps (`001_`, `002_`, `003_`) executed sequentially by the feature's `runner.py`.
3. **Docstrings & Metadata**: Every migration step documents its creation date, purpose, and schema impacts.
4. **Idempotency**: All migrations check existing state, merge non-destructively, and use conditional field deletions. Running any migration multiple times will never crash, duplicate records, or overwrite customized production settings.

### Running Migrations

```bash
# Run migrations for a specific feature (in chronological order)
make migrate smart-model-routing
make migrate user-auth
make migrate speech-to-text
make migrate core-config

# Run all migrations across all features in dependency order
make migrate all

# List all available features, aliases, dates, and migration steps
make migrate-list
```
