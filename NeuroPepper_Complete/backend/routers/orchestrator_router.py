from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"

class PerceptionRequest(BaseModel):
    persons_detected: int = 0
    dominant_emotion: Optional[str] = None
    identified_user: Optional[str] = None
    speech_detected: bool = False
    transcribed_text: Optional[str] = None

@router.post("/chat")
async def orchestrator_chat(request: ChatRequest):
    from backend.agents.orchestrator import orchestrator
    result = await orchestrator.process_input(request.message, request.user_id)
    return result

@router.post("/perception")
async def process_perception(request: PerceptionRequest):
    from backend.agents.orchestrator import orchestrator, PerceptionData
    perception = PerceptionData(
        persons_detected=request.persons_detected,
        dominant_emotion=request.dominant_emotion,
        identified_user=request.identified_user,
        speech_detected=request.speech_detected,
        transcribed_text=request.transcribed_text
    )
    result = await orchestrator.process_perception(perception)
    return result or {"action": "none"}

@router.post("/reset")
async def reset_orchestrator():
    from backend.agents.orchestrator import orchestrator
    orchestrator.reset()
    return {"success": True}

@router.get("/state")
async def get_state():
    from backend.agents.orchestrator import orchestrator
    state = orchestrator.state
    return {
        "robot_state": state.robot_state,
        "current_user": state.current_user_id,
        "messages_count": len(state.messages),
        "last_action": state.last_action,
        "error_count": state.error_count
    }

@router.get("/tools")
async def get_tools():
    from backend.agents.orchestrator import ROBOT_TOOLS
    return {"tools": ROBOT_TOOLS}
