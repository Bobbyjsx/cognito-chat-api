from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.database import create_db_client, init_db
from app.router import auth, chats, config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Firebase on startup
    init_db()
    app.state.db_client = create_db_client()
    yield
    # Any cleanup on shutdown
    # Not strictly necessary to close AsyncClient manually here, but we can if we want.


app = FastAPI(
    title=settings.app_name,
    description="A modular API leveraging the Antigravity AI SDK.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(chats.router)
app.include_router(config.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
