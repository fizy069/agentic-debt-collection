"""OpenAI STT/TTS voice client for the Resolution stage voice layer."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

logger = logging.getLogger(__name__)


class VoiceServiceError(Exception):
    """Raised when an OpenAI STT/TTS call fails."""

    def __init__(self, operation: str, detail: str) -> None:
        self.operation = operation
        self.detail = detail
        super().__init__(f"voice_{operation}: {detail}")


_ALLOWED_AUDIO_MIMES = frozenset(
    {
        "audio/webm",
        "audio/ogg",
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/m4a",
    }
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def is_voice_enabled() -> bool:
    return os.getenv("AGENT2_VOICE_ENABLED", "false").lower() in ("true", "1", "yes")


class VoiceClient:
    """Thin async wrapper around OpenAI audio transcription and speech APIs."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise VoiceServiceError("init", "OPENAI_API_KEY is required for voice")

        self._client = AsyncOpenAI(api_key=api_key)
        self._stt_model = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
        self._tts_model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self._tts_voice = os.getenv("OPENAI_TTS_VOICE", "alloy")

        logger.info(
            "VoiceClient init  stt_model=%s  tts_model=%s  tts_voice=%s",
            self._stt_model,
            self._tts_model,
            self._tts_voice,
        )

    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Run STT on raw audio bytes. Returns the transcribed text."""
        logger.info(
            "voice_stt_start  model=%s  audio_bytes=%d  filename=%s",
            self._stt_model,
            len(audio_bytes),
            filename,
        )
        try:
            result = await self._client.audio.transcriptions.create(
                model=self._stt_model,
                file=(filename, audio_bytes),
            )
        except APIError as exc:
            logger.error("voice_stt_error  model=%s  detail=%s", self._stt_model, exc)
            raise VoiceServiceError("stt", str(exc)) from exc

        text = result.text.strip()
        logger.info(
            "voice_stt_done  model=%s  text_len=%d",
            self._stt_model,
            len(text),
        )
        return text

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        """Run TTS on text. Returns (mp3_bytes, mime_type)."""
        logger.info(
            "voice_tts_start  model=%s  voice=%s  text_len=%d",
            self._tts_model,
            self._tts_voice,
            len(text),
        )
        try:
            response = await self._client.audio.speech.create(
                model=self._tts_model,
                voice=self._tts_voice,
                input=text,
                response_format="mp3",
            )
        except APIError as exc:
            logger.error("voice_tts_error  model=%s  detail=%s", self._tts_model, exc)
            raise VoiceServiceError("tts", str(exc)) from exc

        audio_bytes = response.content
        logger.info(
            "voice_tts_done  model=%s  audio_bytes=%d",
            self._tts_model,
            len(audio_bytes),
        )
        return audio_bytes, "audio/mpeg"
