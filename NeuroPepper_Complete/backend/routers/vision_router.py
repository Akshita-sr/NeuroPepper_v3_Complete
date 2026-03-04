# backend/routers/vision_router.py
# Vision analysis router

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.services import ollama_service

router = APIRouter()


@router.post("/")
async def analyze_image(
    prompt: str = Form(...),
    model: str = Form("ollama/llava:latest"),
    image: UploadFile = File(...)
):
    """Analyze an image with a vision model."""
    image_bytes = await image.read()
    
    provider, model_name = model.split('/', 1)
    
    if provider == "ollama":
        response = await ollama_service.generate_with_image(
            prompt=prompt,
            image_bytes=image_bytes,
            model=model_name
        )
        return {"model": model, "response": response}
    
    raise HTTPException(400, f"Unsupported vision provider: {provider}")
