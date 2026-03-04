# backend/routers/pepper_router.py
# ============================================
# Pepper Robot Control Router
# ============================================
# API endpoints for controlling the Pepper robot
# through the safe action library.
# ============================================

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

from backend.pepper.action_library import (
    robot_actions,
    GestureType,
    EmotionType,
    LEDColor,
    get_action_definitions
)

router = APIRouter()


# ============================================
# REQUEST MODELS
# ============================================

class SpeakRequest(BaseModel):
    """Request to make Pepper speak."""
    text: str = Field(..., max_length=500, description="Text to speak")
    gesture: Optional[str] = Field(None, description="Gesture name")
    emotion: Optional[str] = Field(None, description="Emotion name")
    speed: float = Field(100, ge=50, le=150, description="Speech speed")
    pitch: float = Field(100, ge=50, le=150, description="Voice pitch")


class MoveRequest(BaseModel):
    """Request to move Pepper."""
    x: float = Field(0, description="Forward/backward meters")
    y: float = Field(0, description="Left/right meters")
    theta: float = Field(0, description="Rotation radians")
    speed: Optional[float] = Field(None, description="Movement speed")


class GestureRequest(BaseModel):
    """Request to play a gesture."""
    gesture: str = Field(..., description="Gesture name from GestureType")


class TabletRequest(BaseModel):
    """Request to display on tablet."""
    content: str = Field(..., description="Content to display")
    content_type: str = Field("text", description="text, image, or web")


class EmotionRequest(BaseModel):
    """Request to set emotion."""
    emotion: str = Field(..., description="Emotion name")


class LEDRequest(BaseModel):
    """Request to set LED colors."""
    group: str = Field("eyes", description="LED group: eyes, ears, chest, all")
    color: str = Field(..., description="Color name")


# ============================================
# STATUS ENDPOINTS
# ============================================

@router.get("/status")
async def get_robot_status():
    """
    Get current robot status.
    
    Returns connection state, battery level, and mode.
    """
    # Try to connect if not already connected
    if not robot_actions.connected and not robot_actions.simulation_mode:
        await robot_actions.connect()
    
    if robot_actions.simulation_mode:
        return {
            "connected": False,
            "mode": "simulation",
            "message": "Running in simulation mode (no robot connected)",
            "battery": None
        }
    
    result = await robot_actions.get_status()
    
    return {
        "connected": robot_actions.connected,
        "mode": "live" if robot_actions.connected else "disconnected",
        "battery": result.data.get("battery") if result.data else None,
        "message": result.message
    }


@router.post("/connect")
async def connect_to_robot():
    """Attempt to connect to Pepper robot."""
    connected = await robot_actions.connect()
    
    return {
        "connected": connected,
        "mode": "live" if connected else "simulation",
        "message": "Connected to Pepper" if connected else "Running in simulation mode"
    }


@router.get("/actions")
async def get_available_actions():
    """
    Get list of available robot actions.
    
    Returns action definitions in function-calling format
    for LLM integration.
    """
    return {
        "actions": get_action_definitions(),
        "gestures": [g.name for g in GestureType],
        "emotions": [e.name for e in EmotionType],
        "led_colors": [c.name for c in LEDColor]
    }


# ============================================
# SPEECH ENDPOINTS
# ============================================

@router.post("/speak")
async def speak(request: SpeakRequest):
    """
    Make Pepper speak with optional gesture and emotion.
    """
    # Convert string to enum if provided
    gesture = None
    if request.gesture:
        try:
            gesture = GestureType[request.gesture.upper()]
        except KeyError:
            raise HTTPException(400, f"Invalid gesture: {request.gesture}")
    
    emotion = None
    if request.emotion:
        try:
            emotion = EmotionType[request.emotion.upper()]
        except KeyError:
            raise HTTPException(400, f"Invalid emotion: {request.emotion}")
    
    result = await robot_actions.speak(
        text=request.text,
        gesture=gesture,
        emotion=emotion,
        speed=request.speed,
        pitch=request.pitch
    )
    
    return {
        "success": result.success,
        "message": result.message
    }


@router.post("/say")
async def say_simple(text: str):
    """Simple speak endpoint - just text."""
    result = await robot_actions.speak(text)
    return {"success": result.success, "message": result.message}


# ============================================
# MOVEMENT ENDPOINTS
# ============================================

@router.post("/move")
async def move(request: MoveRequest):
    """
    Move Pepper to a relative position.
    
    Safety limits are enforced automatically.
    """
    result = await robot_actions.move_to(
        x=request.x,
        y=request.y,
        theta=request.theta
    )
    
    return {
        "success": result.success,
        "message": result.message
    }


@router.post("/move/forward")
async def move_forward(distance: float):
    """Move forward by specified meters."""
    result = await robot_actions.move_forward(distance)
    return {"success": result.success, "message": result.message}


@router.post("/move/backward")
async def move_backward(distance: float):
    """Move backward by specified meters."""
    result = await robot_actions.move_backward(distance)
    return {"success": result.success, "message": result.message}


@router.post("/move/rotate")
async def rotate(angle: float):
    """Rotate by specified degrees."""
    result = await robot_actions.rotate(angle)
    return {"success": result.success, "message": result.message}


@router.post("/stop")
async def stop():
    """Emergency stop all movement."""
    result = await robot_actions.stop_movement()
    return {"success": result.success, "message": result.message}


# ============================================
# GESTURE ENDPOINTS
# ============================================

@router.post("/gesture")
async def play_gesture(request: GestureRequest):
    """Play a gesture animation."""
    try:
        gesture = GestureType[request.gesture.upper()]
    except KeyError:
        raise HTTPException(400, f"Invalid gesture: {request.gesture}")
    
    result = await robot_actions.play_gesture(gesture)
    return {"success": result.success, "message": result.message}


@router.post("/wave")
async def wave():
    """Make Pepper wave."""
    result = await robot_actions.wave()
    return {"success": result.success, "message": result.message}


@router.post("/nod")
async def nod():
    """Make Pepper nod yes."""
    result = await robot_actions.nod_yes()
    return {"success": result.success, "message": result.message}


@router.post("/shake")
async def shake():
    """Make Pepper shake head no."""
    result = await robot_actions.shake_no()
    return {"success": result.success, "message": result.message}


# ============================================
# EMOTION & LED ENDPOINTS
# ============================================

@router.post("/emotion")
async def set_emotion(request: EmotionRequest):
    """Set Pepper's emotional expression."""
    try:
        emotion = EmotionType[request.emotion.upper()]
    except KeyError:
        raise HTTPException(400, f"Invalid emotion: {request.emotion}")
    
    result = await robot_actions.set_emotion(emotion)
    return {"success": result.success, "message": result.message}


@router.post("/leds")
async def set_leds(request: LEDRequest):
    """Set LED colors."""
    try:
        color = LEDColor[request.color.upper()]
    except KeyError:
        raise HTTPException(400, f"Invalid color: {request.color}")
    
    result = await robot_actions.set_led_color(request.group, color)
    return {"success": result.success, "message": result.message}


# ============================================
# TABLET ENDPOINTS
# ============================================

@router.post("/tablet")
async def tablet_display(request: TabletRequest):
    """Display content on Pepper's tablet."""
    result = await robot_actions.show_on_tablet(
        content=request.content,
        content_type=request.content_type
    )
    return {"success": result.success, "message": result.message}


@router.post("/tablet/hide")
async def tablet_hide():
    """Hide tablet content."""
    result = await robot_actions.hide_tablet()
    return {"success": result.success, "message": result.message}


# ============================================
# TRACKING ENDPOINTS
# ============================================

@router.post("/track")
async def track(target: str = "face"):
    """Start tracking a target (face, sound, or stop)."""
    result = await robot_actions.look_at(target)
    return {"success": result.success, "message": result.message}


@router.post("/track/stop")
async def stop_tracking():
    """Stop tracking."""
    result = await robot_actions.stop_tracking()
    return {"success": result.success, "message": result.message}


# ============================================
# COMPOUND ACTIONS
# ============================================

@router.post("/greet")
async def greet(user_name: Optional[str] = None):
    """Perform a friendly greeting."""
    result = await robot_actions.greet_user(user_name)
    return {"success": result.success, "message": result.message}


# ============================================
# AUTONOMOUS LIFE CONTROL
# ============================================

@router.post("/autonomous/disable")
async def disable_autonomous():
    """Disable autonomous life for full AI control."""
    result = await robot_actions.disable_autonomous_life()
    return {"success": result.success, "message": result.message}


@router.post("/autonomous/enable")
async def enable_autonomous():
    """Re-enable autonomous life."""
    result = await robot_actions.enable_autonomous_life()
    return {"success": result.success, "message": result.message}
