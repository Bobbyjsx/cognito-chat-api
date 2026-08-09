from contextlib import asynccontextmanager

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.database import create_db_client, init_db
from app.providers.gemini import GeminiProvider
from app.router import attachments, auth, chats, config, stt
from app.tools.registry import ToolRegistry
from app.repositories.attachments import AttachmentRepository
from app.services.attachments import AttachmentService
from app.api.dependencies import get_storage_backend


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.db_client = create_db_client()
    app.state.provider = GeminiProvider(api_key=settings.gemini_api_key)
    registry = ToolRegistry()
    registry.register_defaults()
    app.state.tool_registry = registry
    
    cleanup_task = asyncio.create_task(_cleanup_loop(app))
    
    yield
    cleanup_task.cancel()



app = FastAPI(
    title=settings.app_name,
    description="A modular API leveraging the Antigravity AI SDK.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
