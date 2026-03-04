# backend/services/vision_service.py
# ============================================
# Vision Service - Complete Visual Perception
# ============================================
# Implements:
# - YOLO-World for open-vocabulary object detection
# - MediaPipe for face/hand/pose tracking
# - DeepFace for emotion detection
# - Face recognition for user identification
# ============================================

import os
import io
import asyncio
import base64
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

# Configuration
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n-world.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
ENABLE_FACE_DETECTION = os.getenv("ENABLE_FACE_DETECTION", "true").lower() == "true"
ENABLE_EMOTION_DETECTION = os.getenv("ENABLE_EMOTION_DETECTION", "true").lower() == "true"
ENABLE_POSE_ESTIMATION = os.getenv("ENABLE_POSE_ESTIMATION", "true").lower() == "true"


@dataclass
class DetectedObject:
    """Detected object with bounding box."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]


@dataclass
class DetectedFace:
    """Detected face with attributes."""
    bbox: Tuple[int, int, int, int]
    landmarks: Dict[str, Tuple[int, int]]
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    identity: Optional[str] = None
    identity_confidence: Optional[float] = None


@dataclass
class PoseEstimate:
    """Human pose estimation."""
    landmarks: Dict[str, Tuple[int, int, float]]  # name -> (x, y, confidence)
    gesture: Optional[str] = None


@dataclass
class SceneAnalysis:
    """Complete scene analysis."""
    objects: List[DetectedObject] = field(default_factory=list)
    faces: List[DetectedFace] = field(default_factory=list)
    poses: List[PoseEstimate] = field(default_factory=list)
    description: Optional[str] = None
    dominant_emotion: Optional[str] = None
    person_count: int = 0


# ============================================
# YOLO-WORLD OBJECT DETECTION
# ============================================

class YOLOWorldDetector:
    """YOLO-World open-vocabulary object detection."""
    
    def __init__(self):
        self._model = None
        self._model_path = YOLO_MODEL
        self._confidence = YOLO_CONFIDENCE
    
    async def load_model(self):
        """Load YOLO-World model."""
        if self._model is not None:
            return
        
        try:
            from ultralytics import YOLO
            
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: YOLO(self._model_path)
            )
            print(f"✅ YOLO-World loaded: {self._model_path}")
        except Exception as e:
            print(f"⚠️ YOLO-World not available: {e}")
            self._model = None
    
    async def detect_objects(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
        confidence: Optional[float] = None
    ) -> List[DetectedObject]:
        """Detect objects in image."""
        await self.load_model()
        
        if self._model is None:
            return []
        
        conf = confidence or self._confidence
        
        try:
            loop = asyncio.get_event_loop()
            
            # Set custom classes if provided (open-vocabulary)
            if classes:
                await loop.run_in_executor(
                    None,
                    lambda: self._model.set_classes(classes)
                )
            
            results = await loop.run_in_executor(
                None,
                lambda: self._model.predict(image, conf=conf, verbose=False)
            )
            
            objects = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    objects.append(DetectedObject(
                        label=result.names[int(box.cls[0])],
                        confidence=float(box.conf[0]),
                        bbox=(x1, y1, x2, y2),
                        center=((x1 + x2) // 2, (y1 + y2) // 2)
                    ))
            
            return objects
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    async def detect_custom(
        self,
        image: np.ndarray,
        prompt: str
    ) -> List[DetectedObject]:
        """Detect objects matching a text prompt."""
        classes = [c.strip() for c in prompt.split(",")]
        return await self.detect_objects(image, classes=classes)


# ============================================
# FACE DETECTION & ANALYSIS
# ============================================

class FaceAnalyzer:
    """Face detection, recognition, and emotion analysis."""
    
    def __init__(self):
        self._face_detector = None
        self._emotion_model = None
        self._known_faces: Dict[str, np.ndarray] = {}
    
    async def load_models(self):
        """Load face analysis models."""
        if self._face_detector is not None:
            return
        
        try:
            # MediaPipe face detection
            import mediapipe as mp
            
            loop = asyncio.get_event_loop()
            self._face_detector = await loop.run_in_executor(
                None,
                lambda: mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
            )
            
            self._face_mesh = await loop.run_in_executor(
                None,
                lambda: mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=5,
                    refine_landmarks=True
                )
            )
            
            print("✅ MediaPipe face detection loaded")
        except Exception as e:
            print(f"⚠️ MediaPipe not available: {e}")
        
        if ENABLE_EMOTION_DETECTION:
            try:
                from deepface import DeepFace
                self._emotion_model = DeepFace
                print("✅ DeepFace emotion detection loaded")
            except Exception as e:
                print(f"⚠️ DeepFace not available: {e}")
    
    async def detect_faces(
        self,
        image: np.ndarray,
        analyze_emotion: bool = True,
        identify: bool = False
    ) -> List[DetectedFace]:
        """Detect and analyze faces."""
        await self.load_models()
        
        if self._face_detector is None:
            return []
        
        faces = []
        
        try:
            # Convert BGR to RGB
            rgb_image = image[:, :, ::-1] if len(image.shape) == 3 else image
            
            loop = asyncio.get_event_loop()
            
            # Face detection
            results = await loop.run_in_executor(
                None,
                lambda: self._face_detector.process(rgb_image)
            )
            
            if not results.detections:
                return []
            
            h, w = image.shape[:2]
            
            for detection in results.detections:
                bbox_rel = detection.location_data.relative_bounding_box
                x1 = int(bbox_rel.xmin * w)
                y1 = int(bbox_rel.ymin * h)
                x2 = int((bbox_rel.xmin + bbox_rel.width) * w)
                y2 = int((bbox_rel.ymin + bbox_rel.height) * h)
                
                face = DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    landmarks={}
                )
                
                # Emotion detection
                if analyze_emotion and self._emotion_model:
                    try:
                        face_region = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if face_region.size > 0:
                            analysis = await loop.run_in_executor(
                                None,
                                lambda fr=face_region: self._emotion_model.analyze(
                                    fr,
                                    actions=['emotion', 'age', 'gender'],
                                    enforce_detection=False
                                )
                            )
                            if analysis:
                                result = analysis[0] if isinstance(analysis, list) else analysis
                                face.emotion = result.get('dominant_emotion')
                                face.emotion_confidence = result.get('emotion', {}).get(face.emotion, 0.0)
                                face.age = result.get('age')
                                face.gender = result.get('dominant_gender')
                    except Exception:
                        pass
                
                # Face recognition
                if identify and self._known_faces:
                    face.identity, face.identity_confidence = await self._identify_face(
                        image[y1:y2, x1:x2]
                    )
                
                faces.append(face)
            
            return faces
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    async def _identify_face(
        self,
        face_image: np.ndarray
    ) -> Tuple[Optional[str], Optional[float]]:
        """Identify a face against known faces."""
        try:
            import face_recognition
            
            loop = asyncio.get_event_loop()
            encoding = await loop.run_in_executor(
                None,
                lambda: face_recognition.face_encodings(face_image)
            )
            
            if not encoding:
                return None, None
            
            encoding = encoding[0]
            
            # Compare to known faces
            best_match = None
            best_distance = 1.0
            
            for name, known_encoding in self._known_faces.items():
                distance = await loop.run_in_executor(
                    None,
                    lambda ke=known_encoding: face_recognition.face_distance([ke], encoding)[0]
                )
                if distance < best_distance and distance < 0.6:
                    best_distance = distance
                    best_match = name
            
            if best_match:
                return best_match, 1.0 - best_distance
            
        except Exception as e:
            print(f"Face identification error: {e}")
        
        return None, None
    
    async def register_face(
        self,
        image: np.ndarray,
        name: str
    ) -> bool:
        """Register a face for future recognition."""
        try:
            import face_recognition
            
            loop = asyncio.get_event_loop()
            encodings = await loop.run_in_executor(
                None,
                lambda: face_recognition.face_encodings(image)
            )
            
            if encodings:
                self._known_faces[name] = encodings[0]
                return True
        except Exception as e:
            print(f"Face registration error: {e}")
        
        return False


# ============================================
# POSE ESTIMATION
# ============================================

class PoseEstimator:
    """MediaPipe pose and hand tracking."""
    
    def __init__(self):
        self._pose = None
        self._hands = None
    
    async def load_models(self):
        """Load pose estimation models."""
        if self._pose is not None:
            return
        
        try:
            import mediapipe as mp
            
            loop = asyncio.get_event_loop()
            
            self._pose = await loop.run_in_executor(
                None,
                lambda: mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False
                )
            )
            
            self._hands = await loop.run_in_executor(
                None,
                lambda: mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2
                )
            )
            
            print("✅ MediaPipe pose/hands loaded")
        except Exception as e:
            print(f"⚠️ MediaPipe pose not available: {e}")
    
    async def estimate_pose(
        self,
        image: np.ndarray
    ) -> List[PoseEstimate]:
        """Estimate human pose."""
        await self.load_models()
        
        if self._pose is None:
            return []
        
        poses = []
        
        try:
            rgb_image = image[:, :, ::-1] if len(image.shape) == 3 else image
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._pose.process(rgb_image)
            )
            
            if results.pose_landmarks:
                h, w = image.shape[:2]
                landmarks = {}
                
                landmark_names = [
                    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
                    'right_eye_inner', 'right_eye', 'right_eye_outer',
                    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
                    'left_index', 'right_index', 'left_thumb', 'right_thumb',
                    'left_hip', 'right_hip', 'left_knee', 'right_knee',
                    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
                    'left_foot_index', 'right_foot_index'
                ]
                
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i < len(landmark_names):
                        landmarks[landmark_names[i]] = (
                            int(landmark.x * w),
                            int(landmark.y * h),
                            landmark.visibility
                        )
                
                # Detect gesture
                gesture = self._detect_gesture(landmarks)
                
                poses.append(PoseEstimate(
                    landmarks=landmarks,
                    gesture=gesture
                ))
            
            return poses
            
        except Exception as e:
            print(f"Pose estimation error: {e}")
            return []
    
    def _detect_gesture(self, landmarks: Dict) -> Optional[str]:
        """Detect simple gestures from pose."""
        try:
            # Get key points
            left_wrist = landmarks.get('left_wrist', (0, 0, 0))
            right_wrist = landmarks.get('right_wrist', (0, 0, 0))
            nose = landmarks.get('nose', (0, 0, 0))
            left_shoulder = landmarks.get('left_shoulder', (0, 0, 0))
            right_shoulder = landmarks.get('right_shoulder', (0, 0, 0))
            
            # Wave detection: hand above shoulder level
            if left_wrist[1] < left_shoulder[1] - 50 and left_wrist[2] > 0.5:
                return "waving_left"
            if right_wrist[1] < right_shoulder[1] - 50 and right_wrist[2] > 0.5:
                return "waving_right"
            
            # Hands up
            if (left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]):
                return "hands_up"
            
            # Crossed arms
            mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
            if (abs(left_wrist[0] - mid_x) < 50 and abs(right_wrist[0] - mid_x) < 50 and
                left_wrist[2] > 0.5 and right_wrist[2] > 0.5):
                return "arms_crossed"
            
        except Exception:
            pass
        
        return None


# ============================================
# COMPLETE SCENE ANALYZER
# ============================================

class SceneAnalyzer:
    """Complete scene analysis combining all vision modules."""
    
    def __init__(self):
        self.yolo = YOLOWorldDetector()
        self.face_analyzer = FaceAnalyzer()
        self.pose_estimator = PoseEstimator()
    
    async def analyze(
        self,
        image: np.ndarray,
        detect_objects: bool = True,
        detect_faces: bool = True,
        detect_poses: bool = False,
        object_classes: Optional[List[str]] = None
    ) -> SceneAnalysis:
        """Perform complete scene analysis."""
        analysis = SceneAnalysis()
        
        tasks = []
        
        if detect_objects:
            tasks.append(self.yolo.detect_objects(image, classes=object_classes))
        else:
            tasks.append(asyncio.sleep(0))  # Placeholder
        
        if detect_faces:
            tasks.append(self.face_analyzer.detect_faces(image))
        else:
            tasks.append(asyncio.sleep(0))
        
        if detect_poses:
            tasks.append(self.pose_estimator.estimate_pose(image))
        else:
            tasks.append(asyncio.sleep(0))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Objects
        if detect_objects and not isinstance(results[0], Exception):
            analysis.objects = results[0] or []
        
        # Faces
        if detect_faces and not isinstance(results[1], Exception):
            analysis.faces = results[1] or []
            
            # Get dominant emotion
            emotions = [f.emotion for f in analysis.faces if f.emotion]
            if emotions:
                from collections import Counter
                analysis.dominant_emotion = Counter(emotions).most_common(1)[0][0]
        
        # Poses
        if detect_poses and not isinstance(results[2], Exception):
            analysis.poses = results[2] or []
        
        # Person count
        analysis.person_count = len(analysis.faces) or len([
            o for o in analysis.objects if o.label.lower() == 'person'
        ])
        
        return analysis


# ============================================
# UTILITY FUNCTIONS
# ============================================

def image_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array."""
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    import cv2
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


# ============================================
# SINGLETON INSTANCES
# ============================================

yolo_detector = YOLOWorldDetector()
face_analyzer = FaceAnalyzer()
pose_estimator = PoseEstimator()
scene_analyzer = SceneAnalyzer()
