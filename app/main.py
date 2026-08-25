import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Initialize application root logger with INFO level so all app logs stream to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)

from app.api.dependencies import get_storage_backend
from app.core.config import settings
from app.database import create_db_client, init_db
from app.providers.gemini import GeminiProvider
from app.repositories.attachments import AttachmentRepository
from app.router import attachments, auth, chats, config, stt
from app.services.attachments import AttachmentService
from app.tools.registry import ToolRegistry


async def _cleanup_loop(app: FastAPI):
    while True:
        try:
            # We cleanup attachments older than 24 hours
            before = datetime.now(timezone.utc) - timedelta(hours=24)
            db = app.state.db_client
            provider = app.state.provider
            storage = get_storage_backend()
            repo = AttachmentRepository(db)
            service = AttachmentService(repo, storage, provider)
            count = await service.cleanup_abandoned_temporary(before)
            if count > 0:
                print(f"Cleaned up {count} abandoned temporary attachments.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error during attachment cleanup: {e}")

        await asyncio.sleep(3600)  # Run every hour


from app.core.redis import redis_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.db_client = create_db_client()
    app.state.provider = GeminiProvider(api_key=settings.gemini_api_key)
    registry = ToolRegistry()
    registry.register_defaults()
    app.state.tool_registry = registry

    from app.ai.router import (
        CompositeRequestAnalyzer,
        GeminiFlashLiteAnalyzer,
        HeuristicFallbackAnalyzer,
        SmartModelRouter,
    )

    flash_analyzer = GeminiFlashLiteAnalyzer(api_key=settings.gemini_api_key)
    heuristic_analyzer = HeuristicFallbackAnalyzer()
    composite_analyzer = CompositeRequestAnalyzer(primary_analyzer=flash_analyzer, fallback_analyzer=heuristic_analyzer)
    app.state.smart_router = SmartModelRouter(analyzer=composite_analyzer)

    await redis_cache.connect()
    cleanup_task = asyncio.create_task(_cleanup_loop(app))

    yield
    cleanup_task.cancel()
    await redis_cache.disconnect()


app = FastAPI(
    title=settings.app_name,
    description="A modular API leveraging the Antigravity AI SDK.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_cache_control_header(request: Request, call_next):
    response = await call_next(request)
    if request.method == "GET" and response.status_code == 200 and "Cache-Control" not in response.headers:
        # Cache all GET requests for 60 seconds, allowing stale-while-revalidate for smooth reloads
        response.headers["Cache-Control"] = "private, max-age=60, stale-while-revalidate=60"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chats.router)
app.include_router(config.router)
app.include_router(stt.router)
app.include_router(attachments.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
