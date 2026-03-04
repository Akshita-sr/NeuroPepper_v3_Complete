# backend/services/sound_service.py
# ============================================
# Sound Event Detection Service
# ============================================
# Implements:
# - PANNs/YAMNet for sound event classification
# - Speaker diarization with pyannote
# - Audio level monitoring
# ============================================

import os
import io
import asyncio
import tempfile
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np

# Configuration
ENABLE_SOUND_DETECTION = os.getenv("ENABLE_SOUND_DETECTION", "true").lower() == "true"
ENABLE_SPEAKER_DIARIZATION = os.getenv("ENABLE_SPEAKER_DIARIZATION", "true").lower() == "true"


@dataclass
class SoundEvent:
    """Detected sound event."""
    label: str
    confidence: float
    start_time: float
    end_time: float
    category: str  # e.g., "music", "speech", "nature", "domestic"


@dataclass
class Speaker:
    """Speaker segment from diarization."""
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class AudioAnalysis:
    """Complete audio analysis."""
    sound_events: List[SoundEvent]
    speakers: List[Speaker]
    dominant_sound: Optional[str]
    has_speech: bool
    audio_level_db: float


# ============================================
# YAMNET SOUND EVENT DETECTION
# ============================================

class YAMNetDetector:
    """YAMNet-based sound event detection."""
    
    # YAMNet sound categories
    CATEGORIES = {
        "speech": ["Speech", "Conversation", "Narration"],
        "music": ["Music", "Musical instrument", "Singing"],
        "nature": ["Bird", "Rain", "Wind", "Thunder"],
        "domestic": ["Door", "Alarm", "Telephone", "Television"],
        "transport": ["Car", "Engine", "Train", "Aircraft"],
        "human": ["Laughter", "Crying", "Cough", "Sneeze"],
    }
    
    def __init__(self):
        self._model = None
        self._class_names = None
    
    async def load_model(self):
        """Load YAMNet model."""
        if self._model is not None:
            return
        
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: hub.load('https://tfhub.dev/google/yamnet/1')
            )
            
            # Load class names
            class_map_path = self._model.class_map_path().numpy()
            self._class_names = self._load_class_names(class_map_path.decode('utf-8'))
            
            print("✅ YAMNet loaded")
        except Exception as e:
            print(f"⚠️ YAMNet not available: {e}")
            self._model = None
    
    def _load_class_names(self, path: str) -> List[str]:
        """Load YAMNet class names."""
        try:
            import csv
            with open(path) as f:
                reader = csv.DictReader(f)
                return [row['display_name'] for row in reader]
        except Exception:
            return []
    
    def _get_category(self, label: str) -> str:
        """Get category for a sound label."""
        for category, labels in self.CATEGORIES.items():
            if any(l.lower() in label.lower() for l in labels):
                return category
        return "other"
    
    async def detect_sounds(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        top_k: int = 5
    ) -> List[SoundEvent]:
        """Detect sound events in audio."""
        await self.load_model()
        
        if self._model is None:
            return []
        
        try:
            import tensorflow as tf
            
            # Convert bytes to waveform
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import scipy.signal
                audio_np = scipy.signal.resample(
                    audio_np,
                    int(len(audio_np) * 16000 / sample_rate)
                )
            
            loop = asyncio.get_event_loop()
            scores, embeddings, log_mel = await loop.run_in_executor(
                None,
                lambda: self._model(audio_np)
            )
            
            scores_np = scores.numpy()
            
            # Get top predictions per frame
            events = []
            frame_duration = 0.96  # YAMNet frame size
            
            for frame_idx, frame_scores in enumerate(scores_np):
                top_indices = np.argsort(frame_scores)[-top_k:][::-1]
                
                for idx in top_indices:
                    if idx < len(self._class_names) and frame_scores[idx] > 0.3:
                        label = self._class_names[idx]
                        events.append(SoundEvent(
                            label=label,
                            confidence=float(frame_scores[idx]),
                            start_time=frame_idx * frame_duration,
                            end_time=(frame_idx + 1) * frame_duration,
                            category=self._get_category(label)
                        ))
            
            # Merge consecutive events with same label
            merged = self._merge_events(events)
            
            return merged
            
        except Exception as e:
            print(f"Sound detection error: {e}")
            return []
    
    def _merge_events(self, events: List[SoundEvent]) -> List[SoundEvent]:
        """Merge consecutive events with same label."""
        if not events:
            return []
        
        # Sort by label then start time
        events.sort(key=lambda e: (e.label, e.start_time))
        
        merged = []
        current = None
        
        for event in events:
            if current is None:
                current = event
            elif (event.label == current.label and 
                  event.start_time <= current.end_time + 0.5):
                # Extend current event
                current = SoundEvent(
                    label=current.label,
                    confidence=max(current.confidence, event.confidence),
                    start_time=current.start_time,
                    end_time=event.end_time,
                    category=current.category
                )
            else:
                merged.append(current)
                current = event
        
        if current:
            merged.append(current)
        
        return merged


# ============================================
# SPEAKER DIARIZATION
# ============================================

class SpeakerDiarizer:
    """Speaker diarization using pyannote."""
    
    def __init__(self):
        self._pipeline = None
    
    async def load_model(self):
        """Load diarization pipeline."""
        if self._pipeline is not None:
            return
        
        try:
            from pyannote.audio import Pipeline
            
            loop = asyncio.get_event_loop()
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv("HF_TOKEN")
                )
            )
            
            print("✅ Speaker diarization loaded")
        except Exception as e:
            print(f"⚠️ Speaker diarization not available: {e}")
            self._pipeline = None
    
    async def diarize(
        self,
        audio_bytes: bytes,
        num_speakers: Optional[int] = None
    ) -> List[Speaker]:
        """Perform speaker diarization."""
        await self.load_model()
        
        if self._pipeline is None:
            return []
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            loop = asyncio.get_event_loop()
            
            diarization_kwargs = {}
            if num_speakers:
                diarization_kwargs['num_speakers'] = num_speakers
            
            diarization = await loop.run_in_executor(
                None,
                lambda: self._pipeline(temp_path, **diarization_kwargs)
            )
            
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append(Speaker(
                    speaker_id=speaker,
                    start_time=turn.start,
                    end_time=turn.end,
                    confidence=0.9  # pyannote doesn't provide confidence
                ))
            
            return speakers
            
        finally:
            os.unlink(temp_path)


# ============================================
# AUDIO LEVEL MONITORING
# ============================================

class AudioLevelMonitor:
    """Monitor audio levels."""
    
    @staticmethod
    def calculate_db(audio_bytes: bytes) -> float:
        """Calculate audio level in dB."""
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            
            if len(audio_np) == 0:
                return -100.0
            
            rms = np.sqrt(np.mean(audio_np ** 2))
            
            if rms == 0:
                return -100.0
            
            db = 20 * np.log10(rms / 32768.0)
            return float(db)
            
        except Exception:
            return -100.0
    
    @staticmethod
    def detect_silence(audio_bytes: bytes, threshold_db: float = -40.0) -> bool:
        """Check if audio is silence."""
        db = AudioLevelMonitor.calculate_db(audio_bytes)
        return db < threshold_db


# ============================================
# COMPLETE AUDIO ANALYZER
# ============================================

class AudioAnalyzerService:
    """Complete audio analysis service."""
    
    def __init__(self):
        self.yamnet = YAMNetDetector()
        self.diarizer = SpeakerDiarizer()
        self.level_monitor = AudioLevelMonitor()
    
    async def analyze(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        detect_sounds: bool = True,
        detect_speakers: bool = False
    ) -> AudioAnalysis:
        """Perform complete audio analysis."""
        
        sound_events = []
        speakers = []
        
        # Detect sound events
        if detect_sounds:
            sound_events = await self.yamnet.detect_sounds(audio_bytes, sample_rate)
        
        # Detect speakers
        if detect_speakers:
            speakers = await self.diarizer.diarize(audio_bytes)
        
        # Calculate audio level
        audio_level = self.level_monitor.calculate_db(audio_bytes)
        
        # Determine dominant sound
        dominant_sound = None
        if sound_events:
            # Get event with highest confidence
            best = max(sound_events, key=lambda e: e.confidence)
            dominant_sound = best.label
        
        # Check for speech
        has_speech = any(e.category == "speech" for e in sound_events)
        
        return AudioAnalysis(
            sound_events=sound_events,
            speakers=speakers,
            dominant_sound=dominant_sound,
            has_speech=has_speech,
            audio_level_db=audio_level
        )


# ============================================
# SINGLETON INSTANCES
# ============================================

yamnet_detector = YAMNetDetector()
speaker_diarizer = SpeakerDiarizer()
audio_level_monitor = AudioLevelMonitor()
audio_analyzer = AudioAnalyzerService()


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

async def detect_sound_events(
    audio_bytes: bytes,
    sample_rate: int = 16000
) -> List[SoundEvent]:
    """Convenience function for sound detection."""
    return await yamnet_detector.detect_sounds(audio_bytes, sample_rate)


async def analyze_audio(
    audio_bytes: bytes,
    sample_rate: int = 16000
) -> AudioAnalysis:
    """Convenience function for full audio analysis."""
    return await audio_analyzer.analyze(audio_bytes, sample_rate)
