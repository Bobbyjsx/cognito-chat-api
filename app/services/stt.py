import base64
import logging
from dataclasses import dataclass

from fastapi import HTTPException
from google import genai

from app.core.config import settings
from app.models.config import AppConfigDB
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
    def __init__(self, config_repo: ConfigRepository, client: genai.Client | None = None):
        self.config_repo = config_repo

        if client is not None:
            self.client = client
        elif settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        else:
            self.client = genai.Client()

    async def get_active_config(self) -> AppConfigDB:
        cfg = await self.config_repo.get_config()
        return cfg

    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptionResult:
        """Transcribe audio using the configured Gemini model.

        Args:
            audio_bytes: Raw audio bytes (webm, mp4, ogg, wav, etc.)
            mime_type: MIME type of the audio (e.g. 'audio/webm;codecs=opus')

        Returns:
            A TranscriptionResult with the transcribed text and token usage.
        """
        cfg = await self.get_active_config()

        if not cfg.enable_ai_stt:
            raise HTTPException(
                status_code=403,
                detail="AI speech-to-text is not enabled.",
            )

        model = cfg.stt_model
        audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")

        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=[
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": audio_b64,
                                }
                            },
                            {"text": TRANSCRIPTION_PROMPT},
                        ]
                    }
                ],
            )
            transcript = (response.text or "").strip()
            tokens_used = (
                getattr(response.usage_metadata, "total_token_count", 0)
                if response.usage_metadata
                else 0
            )
            logger.info(
                "STT transcription successful: %d chars, %d tokens",
                len(transcript),
                tokens_used,
            )
            return TranscriptionResult(transcript=transcript, tokens_used=tokens_used)
        except Exception:
            logger.exception("STT transcription failed")
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please try again.",
            ) from None
