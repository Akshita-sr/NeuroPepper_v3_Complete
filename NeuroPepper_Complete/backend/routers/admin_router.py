# backend/routers/admin_router.py
# ============================================
# Admin Dashboard Router
# ============================================

import psutil
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services import ollama_service

router = APIRouter()


def get_size(bytes_val):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


@router.get("/status")
async def get_system_status():
    """
    Get comprehensive system status including:
    - CPU usage and cores
    - Memory usage
    - GPU status (if available)
    - Available AI models
    """
    # CPU info
    cpu_info = {
        "usage": f"{psutil.cpu_percent(interval=0.1)}%",
        "cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False)
    }
    
    # Memory info
    mem = psutil.virtual_memory()
    memory_info = {
        "total": get_size(mem.total),
        "used": get_size(mem.used),
        "available": get_size(mem.available),
        "percent": mem.percent
    }
    
    # GPU info (if available)
    gpu_info = []
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                "name": gpu.name,
                "utilization": f"{gpu.load * 100:.1f}%",
                "memory_used": f"{gpu.memoryUsed:.0f}MB",
                "memory_total": f"{gpu.memoryTotal:.0f}MB",
                "temperature": f"{gpu.temperature}°C"
            })
    except:
        gpu_info = [{"error": "GPU monitoring not available"}]
    
    # Available models
    ollama_models = await ollama_service.list_models()
    models_info = {
        "ollama": ollama_models,
        "total_count": len(ollama_models)
    }
    
    # Ollama health
    ollama_health = await ollama_service.check_server_health()
    
    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "models": models_info,
        "ollama": ollama_health
    }


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
