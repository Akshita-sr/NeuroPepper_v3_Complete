# backend/routers/memory_router.py
# ============================================
# Memory System Router
# ============================================

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from dataclasses import asdict

from backend.core.memory_manager import memory_manager

router = APIRouter()


class StoreMemoryRequest(BaseModel):
    user_id: str
    content: str
    category: str = "fact"
    importance: float = 0.5


class SearchMemoryRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 5
    category: Optional[str] = None


class SetUserNameRequest(BaseModel):
    user_id: str
    name: str


class BuildContextRequest(BaseModel):
    user_id: str
    current_message: Optional[str] = None


@router.post("/store")
async def store_memory(request: StoreMemoryRequest):
    """Store a new memory for a user."""
    memory = await memory_manager.add(
        user_id=request.user_id,
        content=request.content,
        category=request.category,
        importance=request.importance
    )
    return {"success": True, "memory_id": memory.id}


@router.post("/search")
async def search_memories(request: SearchMemoryRequest):
    """Search for relevant memories."""
    memories = await memory_manager.search(
        user_id=request.user_id,
        query=request.query,
        limit=request.limit,
        category=request.category
    )
    return {"count": len(memories), "memories": [asdict(m) for m in memories]}


@router.get("/user/{user_id}")
async def get_all_user_memories(user_id: str, limit: int = 50):
    """Get all memories for a user."""
    memories = await memory_manager.get_all_memories(user_id, limit)
    return {"count": len(memories), "memories": [asdict(m) for m in memories]}


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory."""
    deleted = await memory_manager.delete_memory(memory_id)
    if deleted:
        return {"success": True}
    raise HTTPException(404, "Memory not found")


@router.delete("/user/{user_id}/clear")
async def clear_user_memories(user_id: str):
    """Delete ALL memories for a user."""
    count = await memory_manager.clear_user_memories(user_id)
    return {"success": True, "deleted_count": count}


@router.get("/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get a user's profile."""
    profile = await memory_manager.get_user_profile(user_id)
    if profile:
        return {"success": True, "profile": asdict(profile)}
    return {"success": False, "message": "No profile found"}


@router.post("/profile/name")
async def set_user_name(request: SetUserNameRequest):
    """Set a user's name."""
    await memory_manager.set_user_name(request.user_id, request.name)
    return {"success": True}


@router.post("/context")
async def build_context(request: BuildContextRequest):
    """Build context string for LLM prompts."""
    context = await memory_manager.build_context_for_conversation(
        user_id=request.user_id,
        current_message=request.current_message
    )
    return {"context": context, "has_context": bool(context.strip())}


@router.post("/consolidate/{user_id}")
async def consolidate_memories(user_id: str):
    """Remove duplicate memories."""
    count = await memory_manager.consolidate_memories(user_id)
    return {"success": True, "removed": count}


@router.post("/decay")
async def apply_memory_decay(days_threshold: int = 30):
    """Apply importance decay to old memories."""
    result = await memory_manager.apply_decay(days_threshold)
    return {"success": True, "result": result}
