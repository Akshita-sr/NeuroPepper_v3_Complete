# backend/agents/orchestrator.py
# LangGraph-style Orchestrator for NeuroPepper

import os
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

ENABLE_SAFETY_VALIDATOR = os.getenv("ENABLE_SAFETY_VALIDATOR", "true").lower() == "true"


class RobotState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    PERFORMING_ACTION = "performing_action"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class PerceptionData:
    persons_detected: int = 0
    faces: List[Dict[str, Any]] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    dominant_emotion: Optional[str] = None
    scene_description: Optional[str] = None
    speech_detected: bool = False
    transcribed_text: Optional[str] = None
    voice_emotion: Optional[str] = None
    sound_events: List[str] = field(default_factory=list)
    identified_user: Optional[str] = None
    user_confidence: float = 0.0


@dataclass
class PepperState:
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_user_id: Optional[str] = None
    perception: PerceptionData = field(default_factory=PerceptionData)
    memory_context: str = ""
    relevant_memories: List[Dict] = field(default_factory=list)
    robot_state: str = "idle"
    last_action: Optional[str] = None
    action_result: Optional[Dict] = None
    current_plan: List[Dict] = field(default_factory=list)
    plan_step: int = 0
    safety_validated: bool = True
    safety_message: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None


ROBOT_TOOLS = [
    {"type": "function", "function": {"name": "speak", "description": "Say something", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "gesture": {"type": "string", "enum": ["WAVE", "EXPLAIN", "THINK", "HAPPY", "SAD", "BOW", "YES", "NO"]}, "emotion": {"type": "string", "enum": ["HAPPY", "SAD", "NEUTRAL", "EXCITED", "THINKING"]}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "move_to", "description": "Move robot", "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "theta": {"type": "number"}}, "required": ["x", "y", "theta"]}}},
    {"type": "function", "function": {"name": "gesture", "description": "Perform gesture", "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": ["WAVE", "BOW", "EXPLAIN", "THINK", "YES", "NO", "SHRUG"]}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "show_on_tablet", "description": "Display on tablet", "parameters": {"type": "object", "properties": {"content": {"type": "string"}, "content_type": {"type": "string", "enum": ["text", "web", "image"]}}, "required": ["content"]}}},
    {"type": "function", "function": {"name": "set_emotion", "description": "Set emotion LEDs", "parameters": {"type": "object", "properties": {"emotion": {"type": "string", "enum": ["HAPPY", "SAD", "NEUTRAL", "THINKING", "EXCITED"]}}, "required": ["emotion"]}}},
    {"type": "function", "function": {"name": "query_knowledge", "description": "Search knowledge base", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "remember", "description": "Store user info", "parameters": {"type": "object", "properties": {"fact": {"type": "string"}, "importance": {"type": "number"}}, "required": ["fact"]}}}
]


class SafetyValidator:
    MAX_SPEED = float(os.getenv("MAX_ROBOT_SPEED", "0.3"))
    MAX_DISTANCE = float(os.getenv("MAX_TRAVEL_DISTANCE", "2.0"))
    MAX_ROTATION = float(os.getenv("MAX_ROTATION_ANGLE", "1.57"))
    FORBIDDEN = ["hurt", "attack", "kill", "destroy", "harm"]
    
    @classmethod
    def validate(cls, action: str, params: Dict) -> tuple[bool, str]:
        if action == "speak":
            text = params.get("text", "").lower()
            if any(w in text for w in cls.FORBIDDEN):
                return False, "Harmful content"
            return True, "OK"
        elif action == "move_to":
            x, y = abs(params.get("x", 0)), abs(params.get("y", 0))
            theta = abs(params.get("theta", 0))
            dist = (x**2 + y**2) ** 0.5
            if dist > cls.MAX_DISTANCE:
                return False, f"Distance {dist:.2f}m exceeds max"
            if theta > cls.MAX_ROTATION:
                return False, f"Rotation exceeds max"
            return True, "OK"
        return True, "OK"


async def retrieve_memory(state: PepperState) -> PepperState:
    if not state.current_user_id:
        return state
    try:
        from backend.core.memory_manager import memory_manager
        msg = state.messages[-1]["content"] if state.messages else None
        state.memory_context = await memory_manager.build_context_for_conversation(state.current_user_id, msg)
        if msg:
            memories = await memory_manager.search(state.current_user_id, msg, 5)
            state.relevant_memories = [{"content": m.content, "type": m.memory_type} for m in memories]
    except Exception as e:
        print(f"Memory error: {e}")
    return state


async def think_and_plan(state: PepperState) -> PepperState:
    state.robot_state = "thinking"
    try:
        from backend.services import ollama_service
        
        system = f"""You are Pepper, an intelligent robot assistant.
Perception: {state.perception.persons_detected} persons, emotion: {state.perception.dominant_emotion or 'unknown'}
{state.memory_context}

Available actions: speak, gesture, move_to, show_on_tablet, set_emotion, query_knowledge, remember
Be helpful and natural."""
        
        user_msg = state.messages[-1]["content"] if state.messages else "Hello"
        response = await ollama_service.generate_with_tools(user_msg, ROBOT_TOOLS, system)
        
        if response.get("tool_calls"):
            state.current_plan = [{"action": tc.get("function", {}).get("name"), "params": tc.get("function", {}).get("arguments", {})} for tc in response["tool_calls"]]
        else:
            state.current_plan = [{"action": "speak", "params": {"text": response.get("content", "Hello!")}}]
        state.plan_step = 0
    except Exception as e:
        print(f"Planning error: {e}")
        state.current_plan = [{"action": "speak", "params": {"text": "I had trouble processing that."}}]
    return state


async def validate_safety(state: PepperState) -> PepperState:
    if not ENABLE_SAFETY_VALIDATOR:
        state.safety_validated = True
        return state
    
    issues = []
    for step in state.current_plan:
        valid, msg = SafetyValidator.validate(step.get("action"), step.get("params", {}))
        if not valid:
            issues.append(msg)
    
    state.safety_validated = len(issues) == 0
    if not state.safety_validated:
        state.safety_message = "; ".join(issues)
        state.current_plan = [{"action": "speak", "params": {"text": "I can't do that for safety reasons."}}]
    return state


async def execute_action(state: PepperState) -> PepperState:
    state.robot_state = "performing_action"
    if state.plan_step >= len(state.current_plan):
        state.robot_state = "idle"
        return state
    
    action = state.current_plan[state.plan_step]
    action_name = action.get("action")
    params = action.get("params", {})
    
    try:
        if action_name == "speak":
            state.robot_state = "speaking"
            result = {"success": True, "action": "speak", "text": params.get("text")}
        elif action_name == "query_knowledge":
            from backend.core.rag_manager import rag_manager
            rag_resp = await rag_manager.query(params.get("query", ""))
            result = {"success": True, "action": "query_knowledge", "answer": rag_resp.answer}
        elif action_name == "remember":
            if state.current_user_id:
                from backend.core.memory_manager import memory_manager
                await memory_manager.add(state.current_user_id, params.get("fact", ""), importance=params.get("importance", 0.5))
            result = {"success": True, "action": "remember"}
        else:
            result = {"success": True, "action": action_name}
        state.action_result = result
        state.last_action = action_name
    except Exception as e:
        state.action_result = {"success": False, "error": str(e)}
        state.error_count += 1
        state.last_error = str(e)
    
    state.plan_step += 1
    return state


async def update_memory(state: PepperState) -> PepperState:
    if not state.current_user_id or len(state.messages) < 2:
        return state
    try:
        from backend.core.memory_manager import memory_manager
        await memory_manager.extract_and_store_memories(state.current_user_id, state.messages[-4:])
    except Exception as e:
        print(f"Memory update error: {e}")
    return state


class PepperOrchestrator:
    def __init__(self):
        self.state = PepperState()
    
    async def process_input(self, user_message: str, user_id: Optional[str] = None, perception: Optional[PerceptionData] = None) -> Dict[str, Any]:
        self.state.messages.append({"role": "user", "content": user_message})
        self.state.current_user_id = user_id or "anonymous"
        if perception:
            self.state.perception = perception
        
        self.state = await retrieve_memory(self.state)
        self.state = await think_and_plan(self.state)
        self.state = await validate_safety(self.state)
        
        results = []
        while self.state.plan_step < len(self.state.current_plan):
            self.state = await execute_action(self.state)
            if self.state.action_result:
                results.append(self.state.action_result)
        
        self.state = await update_memory(self.state)
        
        response_text = ""
        for r in results:
            if r.get("action") == "speak":
                response_text = r.get("text", "")
                break
            elif r.get("action") == "query_knowledge":
                response_text = r.get("answer", "")
                break
        
        if response_text:
            self.state.messages.append({"role": "assistant", "content": response_text})
        
        return {"response": response_text, "actions": results, "state": self.state.robot_state, "safety_validated": self.state.safety_validated}
    
    async def process_perception(self, perception: PerceptionData) -> Optional[Dict[str, Any]]:
        self.state.perception = perception
        if perception.persons_detected > 0 and not self.state.messages:
            return await self.process_input(f"[User approached: {perception.identified_user or 'unknown'}]", perception.identified_user)
        if perception.speech_detected and perception.transcribed_text:
            return await self.process_input(perception.transcribed_text, perception.identified_user)
        return None
    
    def reset(self):
        self.state = PepperState()


orchestrator = PepperOrchestrator()
