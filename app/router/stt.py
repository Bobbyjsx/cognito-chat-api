from fastapi import APIRouter, Depends, File, Form, UploadFile
from google.cloud.firestore_v1.async_client import AsyncClient
from pydantic import BaseModel

from app.database import get_db
from app.repositories.config import ConfigRepository
from app.services.stt import STTService

router = APIRouter(prefix="/stt", tags=["stt"])


class TranscribeResponse(BaseModel):
    transcript: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file recorded by the browser"),
    mime_type: str = Form(
        default="audio/webm",
        description="MIME type of the uploaded audio (e.g. audio/webm;codecs=opus)",
    ),
    db: AsyncClient = Depends(get_db),
):
    """Transcribe uploaded audio using the configured AI STT model.

    The client should send a multipart/form-data request with:
    - `audio`: the audio blob recorded via MediaRecorder
    - `mime_type`: the MIME type string (optional, defaults to audio/webm)
    """
    audio_bytes = await audio.read()
    effective_mime = mime_type or audio.content_type or "audio/webm"

    config_repo = ConfigRepository(db)
    service = STTService(config_repo=config_repo)
    transcript = await service.transcribe(audio_bytes, effective_mime)

    return TranscribeResponse(transcript=transcript)
