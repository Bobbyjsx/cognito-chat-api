import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from google.cloud.firestore_v1.async_client import AsyncClient
from pydantic import BaseModel

from app.api.dependencies import get_current_user, get_provider
from app.database import get_db
from app.models.users import UserDB
from app.providers.base import BaseProvider
from app.repositories.config import ConfigRepository
from app.repositories.users import UserRepository
from app.services.stt import STTService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["stt"])


class TranscribeResponse(BaseModel):
    transcript: str
    tokens_used: int = 0


def get_stt_service(
    db: AsyncClient = Depends(get_db),
    provider: BaseProvider = Depends(get_provider),
) -> STTService:
    return STTService(config_repo=ConfigRepository(db), provider=provider)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file recorded by the browser"),
    mime_type: str = Form(
        default="audio/webm",
        description="MIME type of the uploaded audio (e.g. audio/webm;codecs=opus)",
    ),
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
    service: STTService = Depends(get_stt_service),
):
    """Transcribe uploaded audio using the configured AI STT model.

    Token usage from the transcription is attributed to the calling user's
    quota (tokens_used_6h / tokens_used_weekly) after a successful call.

    The client should send a multipart/form-data request with:
    - `audio`: the audio blob recorded via MediaRecorder
    - `mime_type`: the MIME type string (optional, defaults to audio/webm)
    """
    audio_bytes = await audio.read()
    effective_mime = mime_type or audio.content_type or "audio/webm"

    config_repo = ConfigRepository(db)
    cfg = await config_repo.get_config()
    result = await service.transcribe(audio_bytes, effective_mime)

    if result.tokens_used > 0:
        within_limit = await UserRepository(db).atomic_increment_if_within_limit(
            current_user.id,
            result.tokens_used,
            default_limit_6h=cfg.default_token_limit_6h,
            default_limit_weekly=cfg.default_token_limit_weekly,
        )
        logger.info(
            "STT usage user=%s tokens=%d within_limit=%s",
            current_user.id,
            result.tokens_used,
            within_limit,
        )
        if not within_limit:
            raise HTTPException(
                status_code=429,
                detail="Token limit exceeded after transcription. Please upgrade your plan.",
            )

    return TranscribeResponse(
        transcript=result.transcript,
        tokens_used=result.tokens_used,
    )