from fastapi import APIRouter, Depends
from google.cloud.firestore_v1.async_client import AsyncClient

from app.database import get_db
from app.models.config import AppConfigDB
from app.repositories.config import ConfigRepository

router = APIRouter(prefix="/config", tags=["config"])


@router.get("", response_model=AppConfigDB)
async def get_system_config(db: AsyncClient = Depends(get_db)):
    repo = ConfigRepository(db)
    return await repo.get_config()
