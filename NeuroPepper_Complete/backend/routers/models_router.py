# backend/routers/models_router.py
# ============================================
# Available Models Router
# ============================================

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services import ollama_service

router = APIRouter()


@router.get("/")
async def get_all_available_models():
    """
    Get all available AI models organized by category.
    """
    # Get local Ollama models
    ollama_models_raw = await ollama_service.list_models()
    
    # Filter out embedding models from text generation list
    ollama_text_models = [m for m in ollama_models_raw if "embed" not in m.lower()]
    
    # Vision-capable Ollama models
    vision_keywords = ["llava", "bakllava", "moondream", "vision"]
    ollama_vision_models = [
        m for m in ollama_models_raw 
        if any(kw in m.lower() for kw in vision_keywords)
    ]
    
    # Embedding models
    ollama_embedding_models = [
        m for m in ollama_models_raw 
        if "embed" in m.lower() or "nomic" in m.lower()
    ]
    
    # Define remote/cloud models (configure based on available API keys)
    remote_text_models = [
        "openai/gpt-4-turbo",
        "claude/claude-3-haiku-20240307",
        "gemini/gemini-1.5-flash"
    ]
    
    remote_vision_models = [
        "gemini/gemini-1.5-flash-latest"
    ]
    
    remote_image_models = [
        "openai/dall-e-3",
        "stability/sd3"
    ]
    
    local_image_models = [
        "comfyui/sd_xl_base_1.0.safetensors"
    ]
    
    remote_video_models = [
        "runway/gen-2"
    ]
    
    all_models = {
        "ollama_text": ollama_text_models,
        "ollama_vision": ollama_vision_models,
        "ollama_embedding": ollama_embedding_models,
        "ollama": ollama_text_models,  # For backwards compatibility
        "remote_text": remote_text_models,
        "remote_multimodal": remote_vision_models,
        "remote_image": remote_image_models,
        "local_image": local_image_models,
        "remote_video": remote_video_models
    }
    
    return JSONResponse(content=all_models)


@router.get("/recommended")
async def get_recommended_models():
    """
    Get recommended models for different tasks.
    """
    return {
        "text_generation": {
            "primary": "ollama/qwen2.5:7b",
            "fast": "ollama/qwen2.5:3b",
            "description": "Best for general conversations and tasks"
        },
        "vision": {
            "primary": "ollama/llava:latest",
            "description": "Best for image understanding"
        },
        "embedding": {
            "primary": "ollama/nomic-embed-text",
            "description": "Required for RAG document processing"
        },
        "code": {
            "primary": "ollama/codellama:7b",
            "description": "Best for coding tasks"
        }
    }
