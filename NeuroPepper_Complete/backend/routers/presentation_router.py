# backend/routers/presentation_router.py
# ============================================
# Presentation Generation Router
# ============================================

import json
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services import ollama_service

router = APIRouter()


class PresentationRequest(BaseModel):
    topic: str
    model: str = "ollama/qwen2.5:7b"
    num_slides: int = 5


PRESENTATION_PROMPT = """
You are an expert presentation creator. Generate a presentation on: "{topic}"

Create {num_slides} slides. Return ONLY a JSON array (no markdown, no explanation).
Each slide must have: "title", "content" (bullet points as single string with \\n), "speaker_notes"

Start with [ and end with ]
"""


@router.post("/generate")
async def generate_presentation(request: PresentationRequest):
    """
    Generate presentation slides using an LLM.
    """
    provider, model_name = request.model.split('/', 1)
    
    if provider != "ollama":
        raise HTTPException(400, "Only Ollama models supported")
    
    prompt = PRESENTATION_PROMPT.format(
        topic=request.topic,
        num_slides=request.num_slides
    )
    
    response = await ollama_service.generate_text(prompt, model_name)
    
    # Parse JSON from response
    try:
        match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found")
        
        slides = json.loads(match.group(0))
        return {"topic": request.topic, "slides": slides}
        
    except Exception as e:
        raise HTTPException(500, f"Failed to parse slides: {e}")
