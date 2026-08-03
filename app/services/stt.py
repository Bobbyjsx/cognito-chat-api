import logging
from dataclasses import dataclass

from fastapi import HTTPException

from app.models.config import AppConfigDB
from app.providers.base import BaseProvider, classify_provider_error
from app.repositories.config import ConfigRepository

logger = logging.getLogger(__name__)

TRANSCRIPTION_PROMPT = (
    "Transcribe the speech in this audio exactly as spoken. "
    "Return only the transcription text with no additional commentary, "
    "labels, or formatting."
)


@dataclass
class TranscriptionResult:
    transcript: str
    tokens_used: int


class STTService:
    """Speech-to-text built on the shared AI provider.

    Kept for backwards compatibility with the dedicated transcription
    endpoint; the preferred path for audio is now chat attachments (upload the
    audio file and ask questions about it directly).
    """

    def __init__(self, config_repo: ConfigRepository, provider: BaseProvider):
        self.config_repo = config_repo
        self.provider = provider

    async def get_active_config(self) -> AppConfigDB:
        return await self.config_repo.get_config()

    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptionResult:
        """Transcribe audio using the configured Gemini model."""
        cfg = await self.get_active_config()

        if not cfg.enable_ai_stt:
            raise HTTPException(
                status_code=403,
                detail="AI speech-to-text is not enabled.",
            )

        try:
            transcript, tokens_used = await self.provider.transcribe_audio(
                model=cfg.stt_model,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                prompt=TRANSCRIPTION_PROMPT,
            )
            logger.info(
                "STT transcription successful: %d chars, %d tokens",
                len(transcript),
                tokens_used,
            )
            return TranscriptionResult(transcript=transcript, tokens_used=tokens_used)
        except HTTPException:
            raise
        except Exception as exc:
            status, _, message = classify_provider_error(exc)
            logger.exception("STT transcription failed")
            raise HTTPException(
                status_code=status,
                detail=message or "Transcription failed. Please try again.",
            ) from None
