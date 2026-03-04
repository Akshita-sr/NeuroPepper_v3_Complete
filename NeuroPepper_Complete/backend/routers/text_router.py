# backend/routers/text_router.py
# ============================================
# Text Generation Router
# ============================================

import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from backend.services import ollama_service

router = APIRouter()


class TextPromptRequest(BaseModel):
    prompt: str
    models: List[str]
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False


@router.post("/")
async def generate_text(request: TextPromptRequest):
    """
    Generate text from one or more models.
    
    Supports both streaming and non-streaming responses.
    """
    if request.stream and len(request.models) == 1:
        # Single model streaming
        model = request.models[0]
        
        async def stream_generator():
            async for token in ollama_service.generate_text_stream(
                prompt=request.prompt,
                model=model,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
    
    # Non-streaming or multi-model
    tasks = []
    for model_id in request.models:
        provider, model_name = model_id.split('/', 1)
        
        if provider == "ollama":
            tasks.append(ollama_service.generate_text(
                prompt=request.prompt,
                model=model_name,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ))
        else:
            # Handle other providers
            tasks.append(_generate_placeholder(model_id))
    
    if not tasks:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid models selected"}
        )
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    response = {}
    for model_id, result in zip(request.models, results):
        if isinstance(result, Exception):
            response[model_id] = {"error": str(result)}
        else:
            response[model_id] = {"response": result}
    
    return {"prompt": request.prompt, "results": response}


async def _generate_placeholder(model_id: str):
    """Placeholder for unsupported providers."""
    return f"Provider for {model_id} not configured"


@router.post("/stream/{model}")
async def stream_text(model: str, prompt: str, system_prompt: Optional[str] = None):
    """
    Stream text generation for a single model.
    """
    async def stream_generator():
        async for token in ollama_service.generate_text_stream(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )
