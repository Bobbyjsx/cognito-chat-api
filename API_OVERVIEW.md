# Cognito Chat API — Technical Overview

Complete technical overview of the FastAPI backend powering the Cognito chat application. Written for an AI engineer with no prior knowledge of this codebase, so you can implement new AI features without reading the entire repository.

Secrets (`.env`, `firebase-credentials.json`, API keys) are intentionally **not** reproduced here.

> **Note (stale parts):** This overview describes the original system. The
> Gemini tools & attachments evolution superseded parts of it: the Gemini SDK
> calls moved behind `app/providers/gemini.py`, chat history handling and
> token-limiting now live in `app/services/chats.py` + `app/services/context.py`,
> and attachments/storage were added. See `docs/ARCHITECTURE.md` for the
> current architecture.

---

## 1. High-Level Architecture

```
┌─────────────────┐        ┌──────────────────────────────┐        ┌─────────────────┐
│  Next.js FE     │  HTTP  │  FastAPI (this repo)         │  SDK   │  Google Gemini   │
│  (cognito-chat) │ ─────► │  app.main:app                │ ─────► │  (google-genai)  │
│  /api/proxy     │        │  uvicorn on :8001            │        │                  │
└─────────────────┘        └──────────┬───────────────────┘        └─────────────────┘
                                      │ AsyncClient
                                      ▼
                              ┌─────────────────┐
                              │  Firestore      │
                              │  (Google Cloud) │
                              └─────────────────┘
```

- **Framework**: FastAPI (async), run by uvicorn (`app.main:app`). Single process, per-request dependency injection.
- **AI provider**: Google Gemini via the official `google-genai` SDK (`from google import genai`). No LangChain/LlamaIndex abstraction layer.
- **Database**: Firestore via `google-cloud-firestore`'s **async** `AsyncClient`. No SQL, no ORM — Pydantic models map 1:1 to Firestore documents.
- **Auth**: Self-contained JWT (PyJWT) + bcrypt password hashing. No external auth provider.
- **Configuration**: Two sources — env vars (`.env`, loaded at import) for infra secrets, and a Firestore document (`configs/app_config`) as the runtime "source of truth" for model availability, feature toggles, and quota defaults.

### App lifecycle (`app/main.py`)
1. `lifespan` runs `init_db()` (initializes `firebase_admin` with the credentials file, else ADC) then `create_db_client()` (builds the async Firestore client; project id pulled from the firebase app, database from `FIRESTORE_DATABASE` env).
2. The client is stored on `app.state.db_client`; the `get_db` dependency (`app/database.py`) returns it per request.
3. Routers mounted: `auth`, `agent`, `config`, `stt`. Global `CORSMiddleware` allows all origins/methods/headers with credentials.

### Layering convention
`router/` (HTTP only) → `services/` (business logic, AI calls) → `repositories/` (Firestore access) → `models/` (Pydantic). Dependencies are constructed manually in each router via `Depends(get_xxx)` factory functions.

---

## 2. Chat Pipeline

Implemented in `app/services/chats.py` (`AgentService`) and `app/router/chats.py`.

### Non-streaming: `POST /agent/chat`
```
validate_and_resolve_config(model, reasoning)
  → quota pre-check (429 if 6h or weekly window exhausted)
  → create session (title = first 5 words of message) OR load existing + ownership check (404)
  → persist user message
  → build contents from session history + current message
  → client.aio.models.generate_content(model, contents, config)
  → extract response text + total_token_count from usage_metadata
  → persist agent message
  → post-generation atomic quota increment (429 if exceeded)
  → return ChatResponse{session_id, title, response}
```

### Streaming: `POST /agent/chat/stream` (SSE)
Same pipeline, but yields SSE events via an async generator wrapped in `StreamingResponse(media_type="text/event-stream")` with `Cache-Control: no-cache` and `X-Accel-Buffering: no`.

SSE protocol (client must parse `event:` / `data:` lines):

| Event | Payload | Meaning |
|---|---|---|
| `session` | `{session_id, title}` | Emitted first; needed for new-session UI |
| `chunk` | `{type, token}` or tool payload | `type: "text"` normal tokens, `type: "reasoning"` thought tokens, `type: "tool_call"`, `type: "tool_result"` |
| `error` | `{detail, code?}` | Any validation/quota/API error mid-stream (also 401s happen at HTTP level) |
| `done` | `{session_id, title, tokens_used, model, reasoning}` | Terminal event after message persisted |

- Streaming errors before the generator starts are also emitted as `event: error` (HTTP 200) rather than raised.
- Token usage is read from the **last chunk** carrying `usage_metadata.total_token_count`.
- On generation failure, an agent message with `error: "[CODE] message"` is persisted so the UI can show a failed-state bubble.

### Error mapping
`extract_genai_error` (`app/services/chats.py:31`) converts `google.genai.errors.APIError` into `(status, code, message)`:
- HTTP 404 / status `NOT_FOUND` → `MODEL_NOT_FOUND`
- Anything else → `GENERATION_FAILED`
- Non-API exceptions → `500 / GENERATION_FAILED / "Model generation failed. Please try again."` (safe generic message, never leaks internals).

---

## 3. AI Provider Layer

- **Only provider wired**: Google Gemini through `google-genai` (`google.genai`). `genai.Client(api_key=settings.gemini_api_key)` (falls back to ADC/vertex-style defaults if key empty).
- **Model selection** is dynamic: the client receives the model name as a string in every call; no model registry in code beyond what's allowed by Firestore config (see §9).
- **Thinking/reasoning**: `types.ThinkingConfig(thinking_budget, include_thoughts=True)` — budget mapped from reasoning level: `low=1024`, `medium=4096`, `high=8192`, unknown→`2048`; `reasoning="none"` sends no thinking config. Thought tokens surface via `event: reasoning` chunks.
- **Tools**: `types.Tool(code_execution=...)` is attached when `code_execution` is in config `allowed_tools`. Tool calls/results are surfaced as SSE `tool_call` / `tool_result` events (with generated ids) but are **not** re-fed into the model mid-stream — the SDK's built-in tool loop runs server-side on Gemini. `google_search` is listed in the config defaults but not constructed in code.
- **Prompt**: fixed system instruction in `get_base_system_instructions()` (`app/services/chats.py:50`): *"You are Cognito, an advanced AI assistant… Format responses cleanly with Markdown."* (Note: `app/utils/prompts.py` has an older, unused variant.)
- **STT** reuses the same client: audio bytes base64-encoded as `inline_data` + a transcription prompt (`app/services/stt.py`).

---

## 4. Conversation & Memory

- **Persistence**: every exchange is stored in Firestore — a `sessions` document plus a `messages` subcollection (see §7). History is **not** sent in a condensed/rolling form: the full message list is loaded and serialized as `types.Content(role, parts)` and passed to `generate_content` (with the current message appended).
- **Session lifecycle**: sessions are created lazily on first message (title auto-derived from the first 5 words — no LLM call for titles). Deleting is **soft** (`is_deleted: True`); deleted sessions are filtered from all queries and treated as not found.
- **Read state**: `read_status` per session — `"not read"` when the agent replies, `"read"` when the user opens it (`GET /sessions/{id}` auto-marks read; explicit `POST /sessions/{id}/read` also exists).
- **No memory limits**: no context-window trimming, summarization, or token-budgeting of history — the full transcript is sent every turn. This is a known limitation (see §12).
- **Cross-user isolation**: every repository query enforces `user_id` equality (sessions filtered, message reads check ownership). UUID v4 identifiers throughout.

---

## 5. Current Capabilities

| Capability | Status | Where |
|---|---|---|
| Text chat (non-streaming + SSE) | ✅ | `AgentService.process_chat` / `stream_chat` |
| Multi-model selection (6 Gemini models) | ✅ | `POST /agent/chat` `model` field, enforced against config |
| Reasoning/thinking levels (`none`–`high`) | ✅ | `reasoning` field; budgets per level |
| Code execution tool | ✅ | surfaced as SSE tool events |
| Chat history + sessions + search | ✅ | title/message search incl. deep message scan |
| Soft delete, unread badges | ✅ | sessions repo |
| Speech-to-text (Gemini STT, audio upload) | ✅ | `POST /stt/transcribe`; gated by `enable_ai_stt` |
| Token quota (6h / weekly, atomic) | ✅ | transactions + 429s; UI exposes via `/auth/me` |
| Image generation (Imagen models listed in config) | ❌ not implemented | only model names in config |
| Video generation (Veo models listed in config) | ❌ not implemented | only model names in config |
| `google_search` tool | ⚠️ listed in config, not wired in code | — |
| Email verification / password reset flows | ❌ | reset is a plain "set new password" endpoint |
| Streaming quota enforcement during generation | ⚠️ enforced after completion | quota pre-check + post-charge only |

---

## 6. API Endpoints

All JSON unless noted. Auth = `Authorization: Bearer <access_token>`.

### Auth — `app/router/auth.py`
| Method | Path | Body/Query | Notes |
|---|---|---|---|
| POST | `/auth/signup` | `{email, password}` | 201; returns `UserResponse` with quota fields |
| POST | `/auth/login` | `{email, password}` | returns `{access_token, refresh_token, token_type}` |
| POST | `/auth/reset-password` | `{email, new_password}` | **No email verification** |
| GET | `/auth/me` | — | auth; quota summary incl. `pct_6h`, `reset_countdown_6h` |
| POST | `/auth/refresh` | `{refresh_token}` | issues new pair |

### Agent — `app/router/chats.py` (prefix `/agent`)
| Method | Path | Notes |
|---|---|---|
| POST | `/agent/chat` | `{message, model?, reasoning?}`; query param `session_id` to continue |
| POST | `/agent/chat/stream` | same body; SSE response |
| GET | `/agent/sessions?q=` | list w/ optional search (title, last message, deep message scan); newest first |
| GET | `/agent/sessions/{session_id}` | full session + messages; **auto-marks read** |
| DELETE | `/agent/sessions/{session_id}` | soft delete (alias: POST `.../delete`) |
| POST | `/agent/sessions/{session_id}/read` | mark read |

### Config — `app/router/config.py`
| GET | `/config` | public; returns full `AppConfigDB` |

### STT — `app/router/stt.py`
| POST | `/stt/transcribe` | multipart: `audio` (file), `mime_type` (form, default `audio/webm`); auth; returns `{transcript, tokens_used}`; charges quota; 403 if AI STT disabled, 429 if quota exceeded |

### System
| GET | `/health` | `{"status": "ok"}` |

---

## 7. Data Models

Pydantic v2 models (`app/models/`). Firestore stores `model_dump(mode="json")` — datetimes become ISO-8601 **strings**, UUIDs become strings.

### User (`users` collection, `app/models/users.py`)
| Field | Type | Notes |
|---|---|---|
| `id` | UUID | doc id |
| `email` | EmailStr | unique (by convention, not index) |
| `hashed_password` | str | bcrypt |
| `tokens_used` | int | lifetime total |
| `tokens_used_6h` / `token_limit_6h` | int / int? | rolling 6h window; `None` → use global default |
| `reset_at` | datetime | 6h window expiry (set at creation +6h) |
| `tokens_used_weekly` / `token_limit_weekly` | int / int? | rolling weekly window |
| `weekly_reset_at` | datetime | +7d |
| `created_at` / `updated_at` | datetime | |

### Session (`sessions` collection, `app/models/chats.py`)
`id` (UUID, doc id), `user_id` (UUID), `title?`, `is_deleted` (bool, default false), `created_at`, `updated_at`, `last_message_content?`, `last_message_role?`, `read_status` ("read"/"not read"), plus `messages: list[ChatMessageDB]` in the model (populated from the subcollection; **not** stored on the doc).

### ChatMessage (`sessions/{id}/messages` subcollection)
`id` (UUID), `session_id`, `role` ("user"/"agent"), `content`, `error?` (failure-state marker), `created_at`.

### Config (`configs/app_config` doc, `app/models/config.py`)
`AppConfigDB` — see §9. Defaults baked into the model; the stored document is the override.

### Wire schemas
`ChatRequest{message (1–32_000 chars), model?, reasoning?}`, `ChatResponse{session_id, title?, response}`, `ChatSessionSchema` (full), `ChatSessionListSchema` (no messages), `TokenResponse`, `UserResponse` (with computed `pct_6h`, `pct_weekly`, `reset_countdown_*`).

---

## 8. Storage

- **Firestore** only (no cache, no queue). Async client created in `app/database.py:create_db_client`; database name from `FIRESTORE_DATABASE` env (default `(default)`).
- **Collections**: `users`, `sessions` (+ `messages` subcollection per session), `configs`.
- **Timestamps** stored as ISO strings (`mode="json"`). `app/utils/datetime.py:ensure_utc` normalizes string/Timestamp/naive datetimes to UTC-aware; used on every read path (handles Firestore `Timestamp` vs string divergence).
- **Atomicity**: Firestore **transactions** — `UserRepository.atomic_increment_if_within_limit` (`app/repositories/users.py:80`) reads the user doc inside a transaction, resets expired windows (persisting the new reset timestamps), checks both limits, and applies `Increment` transforms (always applied even when over-limit, so users can't dodge charging). Messages are written with a `batch` (message doc + session doc update).
- **Queries** (no composite indexes beyond one declared one):
  - sessions by `user_id ==` then sorted in Python by `updated_at` (declared index `user_id ASC + updated_at DESC` in `firestore.indexes.json` exists but the code sorts in memory — the index is currently unused by the code path).
  - users by `email ==` (`.limit(1)`), users by doc id.
  - Search does a streaming scan of all user sessions + deep message subcollection scan (O(all sessions)) — fine for small data, will not scale.
- **Deployment**: Dockerfile (python:3.12-slim, uvicorn on `$PORT` for Cloud Run); `docker-compose.yml` runs a Firestore **emulator** (port 8080) + the API with watch mode; `Makefile` wraps install/run/lint/test/clean.

---

## 9. Configuration

Two layers:

### Env (`.env`, `app/core/config.py`)
| Var | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | "" | Gemini client key (empty → ADC) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | **mostly unused** — models come from Firestore config |
| `FIREBASE_CREDENTIALS_PATH` | "" | path to service-account JSON (else ADC) |
| `FIRESTORE_DATABASE` | "" | Firestore database id (`(default)` fallback) |
| `SECRET_KEY` | placeholder | JWT signing — must be changed in prod |
| `ALGORITHM` | `HS256` | |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 30 | |
| `REFRESH_TOKEN_EXPIRE_DAYS` | 7 | |

### Runtime (Firestore `configs/app_config`, `AppConfigDB`)
The **authoritative** runtime config — seeded by `scripts/seed_app_config.py` (required: `ConfigRepository.get_config` raises 500 if absent). Read on every chat/STT request (no caching).

Key fields: `default_token_limit_6h` (60_000), `default_token_limit_weekly` (300_000), `enable_text_generation` (true), `enable_image_generation` (false), `enable_video_generation` (false), `enable_ai_stt` (false), `stt_model` (`gemini-3.1-flash-lite`), `allowed_reasoning_levels` (`["none","minimal","low","medium","high"]`), `default_reasoning_level` (`medium`), `default_text_model` (`gemini-3.6-flash`), `models_list` (per-model `{description, enabled, reasoning_modes}` — source of truth; `allowed_text_models` is derived), `allowed_image_models`, `allowed_video_models`, `allowed_tools` (`["google_search","code_execution"]`).

Validation rules in `validate_and_resolve_config` (`app/services/chats.py:81`): text gen disabled → 403; model not in enabled list → 400; reasoning level not in global list **and** not in the model's list → 400. Per-user limits (`token_limit_6h`/`token_limit_weekly` on the user doc) override global defaults via `resolve_user_limits` (`app/services/quota.py`).

---

## 10. Middleware / Cross-Cutting

- **CORS**: wildcard allow-origins with credentials (`app/main.py`) — effectively open; acceptable only behind the FE proxy / a gateway. On the FE, the browser never talks to the API directly: Next.js `src/app/api/proxy/[...path]/route.ts` forwards `/api/proxy/*` → `API_BASE_URL` and injects an Atlas API key header server-side.
- **Auth dependency** (`app/api/dependencies.py:get_current_user`): `OAuth2PasswordBearer(tokenUrl="auth/login")`; decodes JWT, **rejects refresh tokens** (`type == "refresh"`), and always re-fetches the user from Firestore so quota counters are fresh.
- **Token service** (`app/core/security.py`): `create_access_token` (no `type` claim, 30m) / `create_refresh_token` (`type: "refresh"`, 7d); bcrypt hashing/verify run in a thread via `anyio.to_thread` (bcrypt blocks the event loop).
- **Global error handling**: no custom exception handlers; routers catch `ValueError`→404, re-raise `HTTPException`, else 500 with logging (`logger.exception`). Gemini errors are mapped inside services (see §2).
- **Logging**: stdlib `logging` per module; usage/quota events logged with user id + token counts.

---

## 11. Extension Points

Where to plug in new AI features:

1. **New model tiers**: add entries to `models_list` in `AppConfigDB` + re-run `scripts/init_config.py` (merge=True preserves other fields). No code changes — model names are passed straight to the SDK.
2. **New tools**: extend the `tool_list` construction in `validate_and_resolve_config` (today only `code_execution` is built) and add the tool name to `allowed_tools` in config. SSE tool events are already wired (`_extract_stream_events`).
3. **New endpoints/features**: follow the pattern — model in `app/models/`, read/write in `app/repositories/`, logic in `app/services/`, thin route in `app/router/`, register in `app/main.py`.
4. **Feature toggles**: add a boolean to `AppConfigDB` (defaults propagate via `init_config.py` merge); toggle without redeploy.
5. **Image/video generation**: config already lists Imagen/Veo models; implement by calling `client.aio.models.generate_content`/`generate_videos` with those model names + inline/prompt content, mirroring `STTService` (bytes-in, result-out) — quota charging can reuse `atomic_increment_if_within_limit`.
6. **Memory management** (context trimming): hook into `_build_contents` — the single place history is assembled.
7. **Multi-provider support**: swap/abstract the `genai.Client` construction (currently two copies: `AgentService.__init__`, `STTService.__init__`); everything else consumes the SDK's `Content`/`Part`/`GenerateContentConfig` types.
8. **SSE contract**: `_extract_stream_events` is the sole translator of SDK stream chunks → wire events; add new event types there + document in `ChatShell` FE consumer.

---

## 12. Missing Features / Known Gaps

- **No context management**: full raw history sent every turn; no trimming, summarization, or max-history cap (breaks on long conversations and burns quota).
- **No concurrency control on sessions**: no locking/versioning — two parallel sends to one session can interleave messages.
- **Quota enforcement is post-hoc** for generation: quota is charged *after* the model responds (blocking at the start is estimated from stored counters; a user can overrun by sending in parallel — the transaction still counts tokens but returns `within_limit=False` only *after* the fact).
- **Auth hardening**: no email verification, no rate limiting on signup/login/reset, refresh tokens are stateless (not revocable/rotatable on the server), `SECRET_KEY` default placeholder, reset-password has no auth challenge.
- **`google_search` tool listed but unimplemented**; image/video generation toggles exist in config but no endpoints.
- **Read-state semantics**: opening a session auto-marks it read on `GET` (side effect on a GET — mild REST smell).
- **Search is O(all sessions)**: deep scan streams every message subcollection of every session for a user; no Firestore full-text search.
- **`/config` is public** (exposes model lists — fine, but note it).
- **CORS `*` with credentials** — relies entirely on the FE proxy for actual security posture.
- **Client-side STT** (browser Web Speech API) exists in the FE when `enable_ai_stt` is false — the API only supports the Gemini STT path.
- **Model naming drift**: `core/config.py` default `gemini_model` (`gemini-2.5-flash`) is obsolete vs Firestore defaults (`gemini-3.x`); harmless because Firestore wins.

---

## 13. Recommended Implementation Plan (example: "better memory")

For any new feature, the standard sequence:

1. **Config first**: add toggles/parameters to `AppConfigDB` (`app/models/config.py`) with safe defaults; update `scripts/init_config.py` to merge; run it against the target Firestore.
2. **Data layer**: extend/replace `app/repositories/*` with Firestore reads/writes; use `ensure_utc` for datetimes, `Increment`/transactions for atomic counters.
3. **Service layer**: implement the feature in `app/services/` consuming SDK types; reuse `extract_genai_error` for provider errors and the quota/`atomic_increment_if_within_limit` pattern for anything that costs tokens.
4. **API layer**: thin route in the matching `app/router/` file; auth via `get_current_user` unless genuinely public; SSE for anything streaming.
5. **Tests**: unit tests in `tests/` against the emulator (`make test` spins up `firestore` service via docker compose); mock `genai.Client` like `conftest.py:mock_agent`; `freezegun` for window logic; run `ruff check .` before committing.
6. **FE contract**: keep SSE events stable; new fields must survive `snake_case` ↔ `camelCase` transformation on the FE (`lib/axios.ts` interceptors) and the proxy at `src/app/api/proxy/[...path]`.

---

## 14. Code References (map)

| Concern | File |
|---|---|
| App bootstrap, CORS, lifespan, router mount | `app/main.py` |
| Env settings | `app/core/config.py` |
| Firestore client + `get_db` | `app/database.py` |
| JWT/bcrypt | `app/core/security.py` |
| `get_current_user` auth dependency | `app/api/dependencies.py` |
| Auth endpoints | `app/router/auth.py`, `app/services/auth.py` |
| Chat endpoints (REST + SSE) | `app/router/chats.py` |
| Chat service (pipeline, SSE events, error map) | `app/services/chats.py` |
| STT endpoint + service | `app/router/stt.py`, `app/services/stt.py` |
| Config endpoint + repo | `app/router/config.py`, `app/repositories/config.py` |
| Quota logic + response builder | `app/services/quota.py` |
| User repo (quota transactions) | `app/repositories/users.py` |
| Session/message repo | `app/repositories/chats.py` |
| Pydantic models | `app/models/{users,chats,config}.py` |
| datetime normalization | `app/utils/datetime.py` |
| Migration/seed scripts | `scripts/*.py` (run manually, idempotent) |
| Tests (emulator-based) | `tests/{conftest,test_app,test_config,test_token_limits}.py` |
| Deploy/config ops | `Dockerfile`, `docker-compose.yml`, `Makefile`, `.env.example`, `firestore.indexes.json` |
| FE proxy (how clients reach this API) | `cognito-chat/src/app/api/proxy/[...path]/route.ts` |
| FE API client conventions | `cognito-chat/src/lib/axios.ts`, `cognito-chat/src/lib/api-config.ts` |
