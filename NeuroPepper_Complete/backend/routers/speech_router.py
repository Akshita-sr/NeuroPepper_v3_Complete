from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io

router = APIRouter()

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: float = 1.0

@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: Optional[str] = None):
    from backend.services.speech_service import whisper_stt
    audio_bytes = await audio.read()
    result = await whisper_stt.transcribe(audio_bytes, language)
    return {"text": result.text, "language": result.language, "confidence": result.confidence, "duration": result.duration}

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    from backend.services.speech_service import tts_service
    result = await tts_service.synthesize(request.text, request.voice, request.speed)
    return StreamingResponse(io.BytesIO(result.audio_bytes), media_type=f"audio/{result.format}")

@router.post("/vad")
async def voice_activity(audio: UploadFile = File(...)):
    from backend.services.speech_service import silero_vad
    audio_bytes = await audio.read()
    segments = await silero_vad.detect_speech(audio_bytes)
    return {"segments": [{"start": s, "end": e} for s, e in segments], "has_speech": len(segments) > 0}

@router.post("/emotion")
async def voice_emotion(audio: UploadFile = File(...)):
    from backend.services.speech_service import voice_emotion
    audio_bytes = await audio.read()
    emotions = await voice_emotion.detect_emotion(audio_bytes)
    return {"emotions": emotions}
