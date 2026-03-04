# backend/services/speech_service.py
# ============================================
# Speech Service - STT & TTS
# ============================================
# Implements:
# - faster-whisper for speech-to-text
# - Silero VAD for voice activity detection
# - Kokoro/Edge TTS for text-to-speech
# ============================================

import os
import io
import asyncio
import tempfile
import base64
from typing import Optional, Dict, Any, List, Tuple, AsyncGenerator
from dataclasses import dataclass
import numpy as np

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
TTS_MODEL = os.getenv("TTS_MODEL", "edge")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))


@dataclass
class TranscriptionResult:
    """Result from speech transcription."""
    text: str
    language: str
    confidence: float
    duration: float
    segments: List[Dict[str, Any]]


@dataclass
class TTSResult:
    """Result from text-to-speech."""
    audio_bytes: bytes
    format: str
    duration: float
    sample_rate: int


# ============================================
# WHISPER SPEECH-TO-TEXT
# ============================================

class WhisperSTT:
    """faster-whisper based speech-to-text."""
    
    def __init__(self):
        self._model = None
        self._device = WHISPER_DEVICE
        self._compute_type = WHISPER_COMPUTE_TYPE
        self._model_name = WHISPER_MODEL
    
    async def load_model(self):
        """Load the Whisper model."""
        if self._model is not None:
            return
        
        try:
            from faster_whisper import WhisperModel
            
            # Run model loading in thread pool
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type
                )
            )
            print(f"✅ Whisper model loaded: {self._model_name}")
        except ImportError:
            print("⚠️ faster-whisper not installed. Using fallback.")
            self._model = None
        except Exception as e:
            print(f"❌ Failed to load Whisper: {e}")
            self._model = None
    
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        await self.load_model()
        
        if self._model is None:
            return TranscriptionResult(
                text="[Whisper not available]",
                language="unknown",
                confidence=0.0,
                duration=0.0,
                segments=[]
            )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            # Transcribe in thread pool
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    temp_path,
                    language=language,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200
                    )
                )
            )
            
            # Collect segments
            segment_list = []
            full_text = []
            total_confidence = 0.0
            
            for segment in segments:
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
                full_text.append(segment.text.strip())
                total_confidence += segment.avg_logprob
            
            avg_confidence = total_confidence / len(segment_list) if segment_list else 0.0
            
            return TranscriptionResult(
                text=" ".join(full_text),
                language=info.language,
                confidence=avg_confidence,
                duration=info.duration,
                segments=segment_list
            )
            
        finally:
            os.unlink(temp_path)
    
    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        """Transcribe streaming audio."""
        # Buffer audio chunks
        buffer = io.BytesIO()
        
        async for chunk in audio_chunks:
            buffer.write(chunk)
            
            # Process every ~3 seconds of audio (assuming 16kHz, 16-bit)
            if buffer.tell() >= 96000:  # ~3 seconds
                result = await self.transcribe(buffer.getvalue())
                if result.text:
                    yield result.text
                buffer = io.BytesIO()
        
        # Process remaining audio
        if buffer.tell() > 0:
            result = await self.transcribe(buffer.getvalue())
            if result.text:
                yield result.text


# ============================================
# VOICE ACTIVITY DETECTION (Silero VAD)
# ============================================

class SileroVAD:
    """Silero Voice Activity Detection."""
    
    def __init__(self):
        self._model = None
        self._threshold = VAD_THRESHOLD
    
    async def load_model(self):
        """Load the VAD model."""
        if self._model is not None:
            return
        
        try:
            import torch
            
            loop = asyncio.get_event_loop()
            model, utils = await loop.run_in_executor(
                None,
                lambda: torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=True
                )
            )
            
            self._model = model
            self._get_speech_timestamps = utils[0]
            self._collect_chunks = utils[1]
            print("✅ Silero VAD loaded")
            
        except Exception as e:
            print(f"⚠️ Silero VAD not available: {e}")
            self._model = None
    
    async def detect_speech(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000
    ) -> List[Tuple[float, float]]:
        """Detect speech segments in audio."""
        await self.load_model()
        
        if self._model is None:
            return [(0.0, len(audio_bytes) / (sample_rate * 2))]
        
        try:
            import torch
            
            # Convert bytes to tensor
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np)
            
            loop = asyncio.get_event_loop()
            timestamps = await loop.run_in_executor(
                None,
                lambda: self._get_speech_timestamps(
                    audio_tensor,
                    self._model,
                    sampling_rate=sample_rate,
                    threshold=self._threshold
                )
            )
            
            return [(t['start'] / sample_rate, t['end'] / sample_rate) for t in timestamps]
            
        except Exception as e:
            print(f"VAD error: {e}")
            return [(0.0, len(audio_bytes) / (sample_rate * 2))]
    
    async def is_speech(
        self,
        audio_chunk: bytes,
        sample_rate: int = 16000
    ) -> bool:
        """Check if audio chunk contains speech."""
        segments = await self.detect_speech(audio_chunk, sample_rate)
        return len(segments) > 0


# ============================================
# TEXT-TO-SPEECH
# ============================================

class TTSService:
    """Text-to-speech service."""
    
    def __init__(self):
        self._model_type = TTS_MODEL
        self._voice = TTS_VOICE
        self._tts_engine = None
    
    async def load_model(self):
        """Load TTS model."""
        if self._tts_engine is not None:
            return
        
        if self._model_type == "edge":
            # Edge TTS (Microsoft's free TTS)
            try:
                import edge_tts
                self._tts_engine = "edge"
                print("✅ Edge TTS ready")
            except ImportError:
                print("⚠️ edge-tts not installed")
        
        elif self._model_type == "kokoro":
            # Kokoro TTS
            try:
                from TTS.api import TTS
                loop = asyncio.get_event_loop()
                self._tts_engine = await loop.run_in_executor(
                    None,
                    lambda: TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                )
                print("✅ Kokoro TTS loaded")
            except ImportError:
                print("⚠️ TTS not installed, using fallback")
                self._tts_engine = "pyttsx3"
        
        else:
            self._tts_engine = "pyttsx3"
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0
    ) -> TTSResult:
        """Convert text to speech."""
        await self.load_model()
        
        voice = voice or self._voice
        
        if self._tts_engine == "edge":
            return await self._synthesize_edge(text, voice, speed)
        elif isinstance(self._tts_engine, str) and self._tts_engine == "pyttsx3":
            return await self._synthesize_pyttsx3(text, speed)
        else:
            return await self._synthesize_tts(text, speed)
    
    async def _synthesize_edge(
        self,
        text: str,
        voice: str,
        speed: float
    ) -> TTSResult:
        """Synthesize using Edge TTS."""
        import edge_tts
        
        rate = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"
        
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        
        # Collect audio
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        return TTSResult(
            audio_bytes=audio_data.getvalue(),
            format="mp3",
            duration=len(audio_data.getvalue()) / 32000,  # Approximate
            sample_rate=24000
        )
    
    async def _synthesize_pyttsx3(
        self,
        text: str,
        speed: float
    ) -> TTSResult:
        """Synthesize using pyttsx3 (offline)."""
        import pyttsx3
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._pyttsx3_to_file(text, temp_path, speed)
            )
            
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            return TTSResult(
                audio_bytes=audio_bytes,
                format="wav",
                duration=len(audio_bytes) / (44100 * 2),
                sample_rate=44100
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _pyttsx3_to_file(self, text: str, path: str, speed: float):
        """Helper for pyttsx3."""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', int(150 * speed))
        engine.save_to_file(text, path)
        engine.runAndWait()
    
    async def _synthesize_tts(
        self,
        text: str,
        speed: float
    ) -> TTSResult:
        """Synthesize using Coqui TTS."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._tts_engine.tts_to_file(
                    text=text,
                    file_path=temp_path,
                    speed=speed
                )
            )
            
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            return TTSResult(
                audio_bytes=audio_bytes,
                format="wav",
                duration=len(audio_bytes) / (22050 * 2),
                sample_rate=22050
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream TTS audio."""
        # For streaming, use Edge TTS which supports it
        if self._tts_engine == "edge":
            import edge_tts
            communicate = edge_tts.Communicate(text, voice or self._voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        else:
            # Fallback: generate full audio then yield
            result = await self.synthesize(text, voice)
            yield result.audio_bytes


# ============================================
# VOICE EMOTION DETECTION
# ============================================

class VoiceEmotionDetector:
    """Detect emotion from voice audio."""
    
    def __init__(self):
        self._model = None
    
    async def load_model(self):
        """Load emotion detection model."""
        if self._model is not None:
            return
        
        try:
            # Use wav2vec2 for speech emotion
            from transformers import pipeline
            
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    "audio-classification",
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                )
            )
            print("✅ Voice emotion model loaded")
        except Exception as e:
            print(f"⚠️ Voice emotion model not available: {e}")
            self._model = None
    
    async def detect_emotion(
        self,
        audio_bytes: bytes
    ) -> Dict[str, float]:
        """Detect emotion from audio."""
        await self.load_model()
        
        if self._model is None:
            return {"neutral": 1.0}
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._model(temp_path)
            )
            
            return {r["label"]: r["score"] for r in results}
        finally:
            os.unlink(temp_path)


# ============================================
# SINGLETON INSTANCES
# ============================================

whisper_stt = WhisperSTT()
silero_vad = SileroVAD()
tts_service = TTSService()
voice_emotion = VoiceEmotionDetector()


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

async def transcribe_audio(
    audio_bytes: bytes,
    language: Optional[str] = None
) -> TranscriptionResult:
    """Convenience function for transcription."""
    return await whisper_stt.transcribe(audio_bytes, language)


async def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0
) -> TTSResult:
    """Convenience function for TTS."""
    return await tts_service.synthesize(text, voice, speed)


async def detect_voice_activity(
    audio_bytes: bytes,
    sample_rate: int = 16000
) -> List[Tuple[float, float]]:
    """Convenience function for VAD."""
    return await silero_vad.detect_speech(audio_bytes, sample_rate)
