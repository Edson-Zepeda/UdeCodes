from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
DEFAULT_SPEECH_MODEL = os.getenv("ELEVENLABS_SPEECH_MODEL", "eleven_flash_v2_5")
DEFAULT_SFX_MODEL = os.getenv("ELEVENLABS_SFX_MODEL", "eleven_monsters")

_DEFAULT_TIMEOUT = httpx.Timeout(20.0, read=60.0)


def _require_api_key() -> str:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs API key not configured. Set ELEVENLABS_API_KEY environment variable.",
        )
    return ELEVENLABS_API_KEY


async def synthesize_speech(
    text: str,
    *,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    voice_settings: Optional[Dict[str, Any]] = None,
) -> bytes:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Speech text must not be empty.")

    api_key = _require_api_key()
    voice = voice_id or DEFAULT_VOICE_ID
    model = model_id or DEFAULT_SPEECH_MODEL

    payload: Dict[str, Any] = {
        "text": text,
        "model_id": model,
    }
    if voice_settings:
        payload["voice_settings"] = voice_settings

    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice}"
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
    }

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"ElevenLabs speech synthesis failed: {response.text}",
        )

    return response.content


async def generate_sound_effect(
    prompt: str,
    *,
    duration_seconds: float = 0.6,
    model_id: Optional[str] = None,
) -> bytes:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Sound prompt must not be empty.")

    api_key = _require_api_key()
    model = model_id or DEFAULT_SFX_MODEL

    payload: Dict[str, Any] = {
        "text": prompt,
        "duration_seconds": duration_seconds,
        "model_id": model,
    }

    url = f"{ELEVENLABS_API_BASE}/sound-generation"
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
    }

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"ElevenLabs sound generation failed: {response.text}",
        )

    return response.content

