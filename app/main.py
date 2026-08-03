from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.database import create_db_client, init_db
from app.providers.gemini import GeminiProvider
from app.router import attachments, auth, chats, config, stt
from app.tools.registry import ToolRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.db_client = create_db_client()
    app.state.provider = GeminiProvider(api_key=settings.gemini_api_key)
    registry = ToolRegistry()
    registry.register_defaults()
    app.state.tool_registry = registry
    yield


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
