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
from app.core.jwks import prefetch_jwks
from app.database import create_db_client, init_db
from app.repositories.attachments import AttachmentRepository
from app.router import attachments, auth, chats, config, stt, tasks
from app.services.attachments import AttachmentService


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


async def _prewarm_services(app: FastAPI):
    """Background task on startup to pre-warm network connections and caches
    so that the very first user request avoids TLS handshake and cold lookup penalties.
    """
    try:
        from app.repositories.config import ConfigRepository

        # 1. Warm Firestore connection & populate system config cache
        config_repo = ConfigRepository(app.state.db_client)
        await config_repo.get_config()

        # 2. Pre-warm Google API TLS connection pool
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                await client.get("https://generativelanguage.googleapis.com", follow_redirects=True)
            except Exception as exc:
                logger.debug("TLS pool pre-warming connection ignored: %s", exc)
        logger.info("Cold-start dependency pre-warming completed successfully.")
    except Exception as exc:
        logger.debug("Pre-warming background task encountered non-critical error: %s", exc)


from app.core.redis import redis_cache


async def _init_ai_stack(app: FastAPI):
    from app.ai.router import (
        CompositeRequestAnalyzer,
        GeminiFlashLiteAnalyzer,
        HeuristicFallbackAnalyzer,
        SmartModelRouter,
    )
    from app.integrations.cloud_tasks import CloudTasksDispatcher
    from app.providers.gemini import GeminiProvider
    from app.tools.registry import ToolRegistry

    # Initialize Provider
    app.state.provider = GeminiProvider(api_key=settings.gemini_api_key)
    logger.info("Initialized GeminiProvider with provided API key.")

    # Initialize CloudTasksDispatcher
    app.state.tasks_dispatcher = CloudTasksDispatcher(
        project=settings.cloud_tasks_project,
        location=settings.cloud_tasks_location,
        queue=settings.cloud_tasks_queue,
        worker_url=settings.cloud_tasks_worker_url,
        service_account_email=settings.cloud_tasks_service_account_email,
    )
    logger.info("Initialized CloudTasksDispatcher.")

    registry = ToolRegistry()
    registry.register_defaults()
    app.state.tool_registry = registry
    flash_analyzer = GeminiFlashLiteAnalyzer(api_key=settings.gemini_api_key)
    heuristic_analyzer = HeuristicFallbackAnalyzer()
    composite_analyzer = CompositeRequestAnalyzer(
        primary_analyzer=flash_analyzer,
        fallback_analyzer=heuristic_analyzer,
    )
    app.state.smart_router = SmartModelRouter(analyzer=composite_analyzer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    jwks_task = asyncio.create_task(prefetch_jwks())
    init_db()
    app.state.db_client = create_db_client()
    await redis_cache.connect()
    await _init_ai_stack(app)
    cleanup_task = asyncio.create_task(_cleanup_loop(app))
    prewarm_task = asyncio.create_task(_prewarm_services(app))

    yield
    jwks_task.cancel()
    cleanup_task.cancel()
    prewarm_task.cancel()
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
    expose_headers=[
        "X-New-Access-Token",
        "X-New-Refresh-Token",
    ],
)

app.include_router(auth.router)
app.include_router(chats.router)
app.include_router(config.router)
app.include_router(stt.router)
app.include_router(attachments.router)
app.include_router(tasks.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
