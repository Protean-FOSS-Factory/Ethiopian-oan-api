"""
Production-Ready Transcription Provider
"""

import base64
import os
from typing import Optional
from abc import ABC, abstractmethod
from helpers.utils import get_logger
from helpers.langfuse_client import observe, update_current_observation
logger = get_logger(__name__)


# Custom Exceptions
class TranscriptionException(Exception):
    """Base exception for transcription errors"""
    pass


class ModelLoadException(TranscriptionException):
    """Exception raised when model fails to load"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Failed to load model: {model_name}")


class InvalidAudioException(TranscriptionException):
    """Exception raised for invalid audio input"""
    pass


class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers"""

    @abstractmethod
    async def transcribe(self, audio_content: str, lang: str = "en") -> str:
        """
        Transcribe audio content

        Args:
            audio_content: Base64 encoded audio string
            lang: Language code

        Returns:
            str: Transcribed text
        """
        pass

    @abstractmethod
    def validate_audio(self, audio_content: str) -> bool:
        """Validate audio input"""
        pass


class AzureSTTProvider(TranscriptionProvider):
    """Transcription using Azure Speech-to-Text REST API (short audio, <= 60s)."""

    _LANG_MAP = {"en": "en-US", "am": "am-ET", "hi": "hi-IN", "mr": "mr-IN"}

    def __init__(self, api_key: str = None, region: str = None):
        self.api_key = api_key or os.getenv("azure_foundary_api_key")
        self.region = region or os.getenv("azure_foundary_region")
        if not self.api_key or not self.region:
            raise ValueError("Azure STT requires azure_foundary_api_key and azure_foundary_region in env")
        self.endpoint = (
            f"https://{self.region}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
        )
        logger.info(f"✅ Azure STT Provider initialized: region={self.region}")

    def validate_audio(self, audio_content: str) -> bytes:
        if not audio_content:
            raise InvalidAudioException("Audio content is empty")
        try:
            audio_bytes = base64.b64decode(audio_content)
            if len(audio_bytes) > 50 * 1024 * 1024:
                raise InvalidAudioException("Audio too large (max 50MB)")
            if len(audio_bytes) < 100:
                raise InvalidAudioException("Audio data too short")
            return audio_bytes
        except base64.binascii.Error as e:
            raise InvalidAudioException(f"Invalid base64 audio data: {e}")

    def _bcp47(self, lang: str) -> str:
        if "-" in lang:
            return lang
        return self._LANG_MAP.get(lang.lower(), lang)

    @staticmethod
    def _detect_content_type(audio_bytes: bytes) -> str:
        """Pick the Azure-compatible Content-Type from the audio's magic bytes.

        Azure REST short-audio supports audio/wav (PCM) and audio/ogg;codecs=opus.
        WebM/Opus is NOT decoded by the REST endpoint and will return empty text.
        """
        head = audio_bytes[:16]
        if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            return "audio/wav; codecs=audio/pcm; samplerate=16000"
        if head[:4] == b"OggS":
            return "audio/ogg; codecs=opus"
        return "audio/wav; codecs=audio/pcm; samplerate=16000"

    @observe(name="stt.azure", as_type="generation")
    async def transcribe(self, audio_content: str, lang: str = "en") -> str:
        import httpx
        audio_bytes = self.validate_audio(audio_content)
        recognition_lang = self._bcp47(lang)
        content_type = self._detect_content_type(audio_bytes)
        magic = audio_bytes[:8].hex()
        logger.info(
            f"Azure STT request: lang={recognition_lang}, bytes={len(audio_bytes)}, "
            f"content_type={content_type}, magic={magic}"
        )
        update_current_observation(
            metadata={
                "provider": "azure",
                "lang": recognition_lang,
                "audio_bytes": len(audio_bytes),
                "content_type": content_type,
            },
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": content_type,
            "Accept": "application/json",
        }
        params = {"language": recognition_lang, "format": "detailed"}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.endpoint, params=params, headers=headers, content=audio_bytes)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("RecognitionStatus")
                text = (data.get("DisplayText") or "").strip()
                if not text:
                    logger.warning(f"Azure STT empty result: status={status}, body={data}")
                if status and status != "Success":
                    raise TranscriptionException(f"Azure STT status={status} body={data}")
                logger.info(f"Transcription (azure): '{text[:50]}'")
                update_current_observation(output=text)
                return text
        except httpx.HTTPStatusError as e:
            raise TranscriptionException(f"Azure STT HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise TranscriptionException(str(e))


class FasterWhisperTranscriptionProvider(TranscriptionProvider):
    """Transcription using faster-whisper-server (OpenAI-compatible /v1/audio/transcriptions)."""

    def __init__(self, base_url: str = None):
        import httpx  # noqa: F401 — ensure httpx is available
        self.base_url = (base_url or os.getenv("FASTER_WHISPER_URL", "http://localhost:8000")).rstrip('/')
        logger.info(f"✅ FasterWhisper Transcription Provider initialized: {self.base_url}")

    def validate_audio(self, audio_content: str) -> bytes:
        if not audio_content:
            raise InvalidAudioException("Audio content is empty")
        try:
            audio_bytes = base64.b64decode(audio_content)
            if len(audio_bytes) > 50 * 1024 * 1024:
                raise InvalidAudioException("Audio too large (max 50MB)")
            if len(audio_bytes) < 100:
                raise InvalidAudioException("Audio data too short")
            return audio_bytes
        except base64.binascii.Error as e:
            raise InvalidAudioException(f"Invalid base64 audio data: {e}")

    @observe(name="stt.faster_whisper", as_type="generation")
    async def transcribe(self, audio_content: str, lang: str = "en") -> str:
        import httpx
        audio_bytes = self.validate_audio(audio_content)
        whisper_lang = lang.split("-")[0]
        update_current_observation(
            metadata={"provider": "faster_whisper", "lang": whisper_lang, "audio_bytes": len(audio_bytes)},
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={"model": os.getenv("FASTER_WHISPER_MODEL", "Systran/faster-whisper-medium"),
                          "language": whisper_lang}
                )
                resp.raise_for_status()
                text = resp.json().get("text", "").strip()
                logger.info(f"Transcription (faster-whisper): '{text[:50]}'")
                update_current_observation(output=text)
                return text
        except Exception as e:
            raise TranscriptionException(str(e))


# Singleton instance - initialized once at startup
_transcription_provider: Optional[TranscriptionProvider] = None


def get_transcription_provider() -> TranscriptionProvider:
    """
    Get or create transcription provider singleton

    Returns:
        TranscriptionProvider: Transcription provider instance
    """
    global _transcription_provider
    if _transcription_provider is None:
        provider_name = os.getenv("STT_PROVIDER", "faster_whisper").strip().lower()
        if provider_name == "azure":
            _transcription_provider = AzureSTTProvider()
        else:
            _transcription_provider = FasterWhisperTranscriptionProvider()
    return _transcription_provider
