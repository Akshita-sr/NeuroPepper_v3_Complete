# backend/routers/video_router.py
# ============================================
# Video Generation Router
# ============================================

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()


class VideoPromptRequest(BaseModel):
    prompt: str
    models: List[str]


@router.post("/")
async def generate_video(request: VideoPromptRequest):
    """
    Generate videos from text prompts.
    """
    results = {}
    
    for model_id in request.models:
        results[model_id] = {
            "error": "Video generation requires RunwayML API key"
        }
    
    return {"prompt": request.prompt, "results": results}
