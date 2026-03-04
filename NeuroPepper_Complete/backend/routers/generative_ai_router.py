# backend/routers/generative_ai_router.py
# ============================================
# Generative AI Router
# ============================================
# API endpoints for Stability AI, Runway ML, and ComfyUI.
# ============================================

import base64
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

from backend.services.generative_ai_service import (
    stability_service,
    runway_service,
    comfyui_service,
    check_providers,
    generate_image,
    generate_video,
)

router = APIRouter()


@router.get("/providers")
async def get_available_providers():
    """Check which generative AI providers are available."""
    providers = await check_providers()
    return {
        "providers": providers,
        "help": {
            "stability_ai": "Add STABILITY_API_KEY to .env (get key at platform.stability.ai)",
            "runway_ml": "Add RUNWAY_API_KEY to .env (get key at dev.runwayml.com)",
            "comfyui_local": "Start ComfyUI with run_nvidia_gpu.bat (free, uses your GPU)",
        }
    }


@router.post("/image/generate")
async def api_generate_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    provider: str = Form("auto"),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(30),
    cfg_scale: float = Form(7.0),
    seed: int = Form(0),
):
    """
    Generate an image from text.
    
    Providers: "auto" (tries free ComfyUI first), "stability", "comfyui"
    """
    result = await generate_image(
        prompt=prompt,
        provider=provider,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )

    if result.success:
        return {
            "success": True,
            "provider": result.provider,
            "image_base64": result.base64_data,
            "file_path": result.file_path,
            "latency_seconds": result.latency_seconds,
            "metadata": result.metadata,
        }
    else:
        return JSONResponse(
            status_code=400 if "not set" in (result.error or "") else 500,
            content={"success": False, "error": result.error, "provider": result.provider},
        )


@router.post("/image/transform")
async def api_image_to_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    strength: float = Form(0.7),
    provider: str = Form("stability"),
):
    """Transform an existing image based on a text prompt (Stability AI only)."""
    # Save uploaded image temporarily
    temp_path = Path("data/generated_media") / f"upload_{image.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await image.read())

    result = await stability_service.image_to_image(
        prompt=prompt, image_path=str(temp_path), strength=strength,
    )

    if result.success:
        return {
            "success": True,
            "provider": result.provider,
            "image_base64": result.base64_data,
            "file_path": result.file_path,
            "latency_seconds": result.latency_seconds,
        }
    else:
        return JSONResponse(status_code=500, content={"success": False, "error": result.error})


@router.post("/video/generate")
async def api_generate_video(
    prompt: str = Form(...),
    duration: int = Form(5),
    model: str = Form("gen3a_turbo"),
):
    """
    Generate a video from text using Runway ML.
    
    Duration: 5 or 10 seconds.
    Models: "gen3a_turbo" (fast) or "gen4_turbo" (newest).
    Note: This can take 1-5 minutes to complete.
    """
    result = await generate_video(prompt=prompt, duration=duration, model=model)

    if result.success:
        return {
            "success": True,
            "provider": result.provider,
            "file_path": result.file_path,
            "latency_seconds": result.latency_seconds,
            "metadata": result.metadata,
        }
    else:
        return JSONResponse(status_code=500, content={"success": False, "error": result.error})


@router.post("/video/animate")
async def api_image_to_video(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    duration: int = Form(5),
    model: str = Form("gen3a_turbo"),
):
    """Animate a still image into a video using Runway ML."""
    temp_path = Path("data/generated_media") / f"upload_{image.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await image.read())

    result = await runway_service.image_to_video(
        prompt=prompt, image_path=str(temp_path), duration=duration, model=model,
    )

    if result.success:
        return {
            "success": True,
            "provider": result.provider,
            "file_path": result.file_path,
            "latency_seconds": result.latency_seconds,
        }
    else:
        return JSONResponse(status_code=500, content={"success": False, "error": result.error})


@router.get("/comfyui/checkpoints")
async def list_comfyui_checkpoints():
    """List available model checkpoints in your local ComfyUI."""
    available = await comfyui_service.is_available()
    if not available:
        return {
            "available": False,
            "checkpoints": [],
            "help": "Start ComfyUI first: run run_nvidia_gpu.bat in your ComfyUI_windows_portable folder",
        }

    checkpoints = await comfyui_service.list_checkpoints()
    return {"available": True, "checkpoints": checkpoints}


@router.get("/stability/models")
async def list_stability_models():
    """List available Stability AI models."""
    return {
        "available": stability_service.is_available(),
        "models": await stability_service.list_available_models(),
    }
