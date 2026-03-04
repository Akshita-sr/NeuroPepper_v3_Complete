# backend/pepper/action_library.py
# ============================================
# Pepper Robot Action Library
# ============================================
# Safe, validated robot actions that the AI can call.
# 
# This provides a whitelist of allowed actions with
# parameter validation to ensure safe robot operation.
# ============================================

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import httpx

from dotenv import load_dotenv
load_dotenv()

# Robot connection settings
PEPPER_IP = os.getenv("PEPPER_IP", "192.168.1.100")
PEPPER_PORT = int(os.getenv("PEPPER_PORT", "9559"))
PEPPER_BRIDGE_URL = f"http://{PEPPER_IP}:{PEPPER_PORT}"

# Safety limits
MAX_ROBOT_SPEED = float(os.getenv("MAX_ROBOT_SPEED", "0.3"))
MAX_TRAVEL_DISTANCE = float(os.getenv("MAX_TRAVEL_DISTANCE", "2.0"))
ENABLE_SAFETY = os.getenv("ENABLE_SAFETY_VALIDATOR", "true").lower() == "true"


class EmotionType(str, Enum):
    """Supported emotional expressions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    THINKING = "thinking"
    EXCITED = "excited"


class GestureType(str, Enum):
    """Available gesture animations."""
    # Greetings
    WAVE = "animations/Stand/Gestures/Hey_1"
    BOW = "animations/Stand/Gestures/BowShort_1"
    
    # Emotions
    HAPPY = "animations/Stand/Emotions/Positive/Happy_1"
    SAD = "animations/Stand/Emotions/Negative/Sad_1"
    EXCITED = "animations/Stand/Emotions/Positive/Excited_1"
    
    # Conversational
    EXPLAIN = "animations/Stand/Gestures/Explain_1"
    THINK = "animations/Stand/Gestures/Think_1"
    SHOW_TABLET = "animations/Stand/Gestures/ShowTablet_1"
    
    # Reactions
    YES = "animations/Stand/Gestures/Yes_1"
    NO = "animations/Stand/Gestures/No_1"
    MAYBE = "animations/Stand/Gestures/IDontKnow_1"
    
    # Waiting
    SCRATCH_HEAD = "animations/Stand/Waiting/ScratchHead_1"
    THINK_WAIT = "animations/Stand/Waiting/Think_1"


class LEDColor(str, Enum):
    """LED color presets."""
    WHITE = "white"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    CYAN = "cyan"
    MAGENTA = "magenta"
    OFF = "off"


@dataclass
class ActionResult:
    """Result of a robot action."""
    success: bool
    action: str
    message: str
    data: Optional[Dict[str, Any]] = None


class RobotActionLibrary:
    """
    Safe action library for Pepper robot control.
    
    All actions are validated before execution to ensure
    the robot operates within safe parameters.
    
    Usage:
        robot = RobotActionLibrary()
        
        # Make robot speak with gesture
        result = await robot.speak(
            "Hello! How can I help you?",
            gesture=GestureType.WAVE,
            emotion=EmotionType.HAPPY
        )
        
        # Move safely
        result = await robot.move_forward(distance=0.5)
    """
    
    def __init__(self):
        """Initialize the action library."""
        self.connected = False
        self.simulation_mode = False
        
    async def connect(self) -> bool:
        """
        Establish connection with Pepper robot.
        
        Returns True if connected, False if in simulation mode.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{PEPPER_BRIDGE_URL}/status")
                response.raise_for_status()
                self.connected = True
                self.simulation_mode = False
                return True
        except Exception:
            print(f"⚠️ Could not connect to Pepper at {PEPPER_IP}:{PEPPER_PORT}")
            print("   Running in simulation mode")
            self.connected = False
            self.simulation_mode = True
            return False
    
    async def _send_command(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> ActionResult:
        """
        Send a command to the robot bridge server.
        """
        if self.simulation_mode:
            # Return simulated success
            return ActionResult(
                success=True,
                action=endpoint,
                message=f"[SIMULATION] {endpoint} executed with {params}",
                data=params
            )
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{PEPPER_BRIDGE_URL}/{endpoint}",
                    json=params
                )
                response.raise_for_status()
                data = response.json()
                
                return ActionResult(
                    success=data.get("success", True),
                    action=endpoint,
                    message=data.get("message", "Command executed"),
                    data=data
                )
        except Exception as e:
            return ActionResult(
                success=False,
                action=endpoint,
                message=f"Error: {str(e)}"
            )
    
    # ============================================
    # SPEECH ACTIONS
    # ============================================
    
    async def speak(
        self,
        text: str,
        gesture: Optional[GestureType] = None,
        emotion: Optional[EmotionType] = None,
        speed: float = 100,
        pitch: float = 100
    ) -> ActionResult:
        """
        Make Pepper speak with optional gesture and emotion.
        
        Args:
            text: What to say (max 500 characters)
            gesture: Optional gesture animation
            emotion: Optional emotional expression (affects LEDs)
            speed: Speech speed (50-150, default 100)
            pitch: Voice pitch (50-150, default 100)
        
        Returns:
            ActionResult indicating success/failure
        """
        # Validate parameters
        if len(text) > 500:
            text = text[:497] + "..."
        
        speed = max(50, min(150, speed))
        pitch = max(50, min(150, pitch))
        
        # Build animated speech text with gesture tags
        animated_text = text
        if gesture:
            animated_text = f"^start({gesture.value}) {text}"
        
        params = {
            "text": animated_text,
            "speed": speed,
            "pitch": pitch
        }
        
        # Set LED color based on emotion
        if emotion:
            await self.set_emotion(emotion)
        
        return await self._send_command("speak", params)
    
    async def say_animated(
        self,
        text: str,
        animation_mode: str = "contextual"
    ) -> ActionResult:
        """
        Speak with automatic gesture generation.
        
        Pepper will choose appropriate gestures based on the text.
        
        Args:
            text: What to say
            animation_mode: "contextual" (default), "random", or "disabled"
        """
        params = {
            "text": text,
            "mode": animation_mode
        }
        return await self._send_command("animated_speech", params)
    
    # ============================================
    # MOVEMENT ACTIONS (with safety validation)
    # ============================================
    
    async def move_forward(
        self,
        distance: float,
        speed: Optional[float] = None
    ) -> ActionResult:
        """
        Move Pepper forward by specified distance.
        
        Args:
            distance: Distance in meters (max MAX_TRAVEL_DISTANCE)
            speed: Movement speed (max MAX_ROBOT_SPEED)
        """
        # Safety validation
        if not ENABLE_SAFETY:
            pass  # Skip validation
        else:
            if distance > MAX_TRAVEL_DISTANCE:
                return ActionResult(
                    success=False,
                    action="move_forward",
                    message=f"Safety limit: Max distance is {MAX_TRAVEL_DISTANCE}m"
                )
            if speed and speed > MAX_ROBOT_SPEED:
                speed = MAX_ROBOT_SPEED
        
        params = {
            "x": distance,
            "y": 0,
            "theta": 0,
            "speed": speed or MAX_ROBOT_SPEED
        }
        return await self._send_command("move", params)
    
    async def move_backward(
        self,
        distance: float,
        speed: Optional[float] = None
    ) -> ActionResult:
        """Move Pepper backward."""
        return await self.move_forward(-distance, speed)
    
    async def rotate(
        self,
        angle: float,
        speed: Optional[float] = None
    ) -> ActionResult:
        """
        Rotate Pepper by specified angle.
        
        Args:
            angle: Rotation in degrees (-180 to 180)
            speed: Rotation speed
        """
        import math
        
        # Convert degrees to radians
        theta = math.radians(max(-180, min(180, angle)))
        
        params = {
            "x": 0,
            "y": 0,
            "theta": theta,
            "speed": speed or 0.2
        }
        return await self._send_command("move", params)
    
    async def move_to(
        self,
        x: float,
        y: float,
        theta: float = 0
    ) -> ActionResult:
        """
        Move to a position relative to current position.
        
        Args:
            x: Forward/backward distance (meters)
            y: Left/right distance (meters)
            theta: Rotation (radians)
        """
        # Safety check
        import math
        total_distance = math.sqrt(x**2 + y**2)
        
        if ENABLE_SAFETY and total_distance > MAX_TRAVEL_DISTANCE:
            return ActionResult(
                success=False,
                action="move_to",
                message=f"Safety limit: Max distance is {MAX_TRAVEL_DISTANCE}m"
            )
        
        params = {
            "x": x,
            "y": y,
            "theta": theta
        }
        return await self._send_command("move_to", params)
    
    async def stop_movement(self) -> ActionResult:
        """Emergency stop all movement."""
        return await self._send_command("stop", {})
    
    # ============================================
    # GESTURE ACTIONS
    # ============================================
    
    async def play_gesture(
        self,
        gesture: GestureType
    ) -> ActionResult:
        """
        Play a specific gesture animation.
        """
        params = {"animation": gesture.value}
        return await self._send_command("gesture", params)
    
    async def wave(self) -> ActionResult:
        """Make Pepper wave."""
        return await self.play_gesture(GestureType.WAVE)
    
    async def nod_yes(self) -> ActionResult:
        """Make Pepper nod yes."""
        return await self.play_gesture(GestureType.YES)
    
    async def shake_no(self) -> ActionResult:
        """Make Pepper shake head no."""
        return await self.play_gesture(GestureType.NO)
    
    async def shrug(self) -> ActionResult:
        """Make Pepper shrug."""
        return await self.play_gesture(GestureType.MAYBE)
    
    # ============================================
    # LED & EMOTION ACTIONS
    # ============================================
    
    async def set_led_color(
        self,
        group: str,
        color: LEDColor
    ) -> ActionResult:
        """
        Set LED color for a specific group.
        
        Args:
            group: "eyes", "ears", "chest", "all"
            color: Color preset
        """
        params = {
            "group": group,
            "color": color.value
        }
        return await self._send_command("leds", params)
    
    async def set_emotion(
        self,
        emotion: EmotionType
    ) -> ActionResult:
        """
        Set Pepper's emotional expression via LEDs.
        
        Maps emotions to eye colors:
        - HAPPY: Green
        - SAD: Blue
        - SURPRISED: Cyan
        - THINKING: White (cycling)
        - EXCITED: Yellow
        - NEUTRAL: White
        """
        emotion_colors = {
            EmotionType.HAPPY: LEDColor.GREEN,
            EmotionType.SAD: LEDColor.BLUE,
            EmotionType.SURPRISED: LEDColor.CYAN,
            EmotionType.THINKING: LEDColor.WHITE,
            EmotionType.EXCITED: LEDColor.YELLOW,
            EmotionType.NEUTRAL: LEDColor.WHITE,
        }
        
        color = emotion_colors.get(emotion, LEDColor.WHITE)
        return await self.set_led_color("eyes", color)
    
    # ============================================
    # TABLET ACTIONS
    # ============================================
    
    async def show_on_tablet(
        self,
        content: str,
        content_type: str = "text"
    ) -> ActionResult:
        """
        Display content on Pepper's tablet.
        
        Args:
            content: Text or URL to display
            content_type: "text", "image", "web"
        """
        params = {
            "content": content,
            "type": content_type
        }
        return await self._send_command("tablet", params)
    
    async def show_image(self, image_url: str) -> ActionResult:
        """Display an image on the tablet."""
        return await self.show_on_tablet(image_url, "image")
    
    async def show_web(self, url: str) -> ActionResult:
        """Display a webpage on the tablet."""
        return await self.show_on_tablet(url, "web")
    
    async def hide_tablet(self) -> ActionResult:
        """Clear the tablet display."""
        return await self._send_command("tablet", {"action": "hide"})
    
    # ============================================
    # TRACKING & ATTENTION
    # ============================================
    
    async def look_at(
        self,
        target: str = "face"
    ) -> ActionResult:
        """
        Make Pepper look at a target.
        
        Args:
            target: "face" (track faces), "sound" (track sounds), "stop"
        """
        params = {"target": target}
        return await self._send_command("track", params)
    
    async def stop_tracking(self) -> ActionResult:
        """Stop any active tracking."""
        return await self.look_at("stop")
    
    # ============================================
    # STATUS & SENSORS
    # ============================================
    
    async def get_status(self) -> ActionResult:
        """Get robot status (battery, temperature, etc.)."""
        return await self._send_command("status", {})
    
    async def get_battery_level(self) -> Optional[int]:
        """Get battery level as percentage."""
        result = await self.get_status()
        if result.success and result.data:
            return result.data.get("battery")
        return None
    
    # ============================================
    # AUTONOMOUS LIFE CONTROL
    # ============================================
    
    async def disable_autonomous_life(self) -> ActionResult:
        """
        Disable autonomous life for full control.
        
        IMPORTANT: Call this before custom AI control.
        """
        return await self._send_command("autonomous_life", {"state": "disabled"})
    
    async def enable_autonomous_life(self) -> ActionResult:
        """Re-enable autonomous life."""
        return await self._send_command("autonomous_life", {"state": "solitary"})
    
    async def set_breathing(self, enabled: bool) -> ActionResult:
        """Enable/disable breathing animation."""
        params = {"enabled": enabled}
        return await self._send_command("breathing", params)
    
    # ============================================
    # COMPOUND ACTIONS
    # ============================================
    
    async def greet_user(self, user_name: Optional[str] = None) -> ActionResult:
        """
        Perform a friendly greeting sequence.
        """
        greeting = f"Hello {user_name}!" if user_name else "Hello there!"
        
        # Set happy emotion
        await self.set_emotion(EmotionType.HAPPY)
        
        # Wave and speak
        return await self.speak(
            f"{greeting} It's nice to see you.",
            gesture=GestureType.WAVE,
            emotion=EmotionType.HAPPY
        )
    
    async def express_confusion(self) -> ActionResult:
        """Express that Pepper is confused."""
        await self.set_emotion(EmotionType.THINKING)
        return await self.speak(
            "Hmm, I'm not quite sure I understand.",
            gesture=GestureType.MAYBE
        )
    
    async def express_understanding(self) -> ActionResult:
        """Express that Pepper understood."""
        await self.set_emotion(EmotionType.HAPPY)
        return await self.speak(
            "I understand!",
            gesture=GestureType.YES
        )


# Create singleton instance
robot_actions = RobotActionLibrary()


# ============================================
# ACTION DEFINITIONS FOR LLM
# ============================================
# These are the function definitions that get passed to the LLM
# for tool calling / function calling

def get_action_definitions() -> List[Dict[str, Any]]:
    """
    Get action definitions in OpenAI function format.
    
    These can be passed to LLMs for tool calling.
    """
    return [
        {
            "name": "speak",
            "description": "Make the robot speak with optional gesture",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "What the robot should say (max 500 chars)"
                    },
                    "gesture": {
                        "type": "string",
                        "enum": [g.name for g in GestureType],
                        "description": "Optional gesture to perform while speaking"
                    },
                    "emotion": {
                        "type": "string",
                        "enum": [e.name for e in EmotionType],
                        "description": "Emotional expression (affects LED colors)"
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "move_forward",
            "description": f"Move the robot forward (max {MAX_TRAVEL_DISTANCE}m)",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance": {
                        "type": "number",
                        "description": "Distance in meters"
                    }
                },
                "required": ["distance"]
            }
        },
        {
            "name": "rotate",
            "description": "Rotate the robot",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "Angle in degrees (-180 to 180)"
                    }
                },
                "required": ["angle"]
            }
        },
        {
            "name": "play_gesture",
            "description": "Play a gesture animation",
            "parameters": {
                "type": "object",
                "properties": {
                    "gesture": {
                        "type": "string",
                        "enum": [g.name for g in GestureType],
                        "description": "The gesture to perform"
                    }
                },
                "required": ["gesture"]
            }
        },
        {
            "name": "show_on_tablet",
            "description": "Display content on the robot's tablet",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Text or URL to display"
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["text", "image", "web"],
                        "description": "Type of content"
                    }
                },
                "required": ["content"]
            }
        },
        {
            "name": "set_emotion",
            "description": "Set the robot's emotional expression (LED colors)",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": [e.name for e in EmotionType],
                        "description": "The emotion to express"
                    }
                },
                "required": ["emotion"]
            }
        },
        {
            "name": "look_at",
            "description": "Make the robot track a target",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["face", "sound", "stop"],
                        "description": "What to track"
                    }
                },
                "required": ["target"]
            }
        }
    ]
