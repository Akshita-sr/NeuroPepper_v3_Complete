# backend/app.py
# ============================================
# NeuroPepper - Main Application
# ============================================
# This is the heart of the NeuroPepper system.
# It connects all the AI services and robot controls.
# ============================================

# --- Core Imports ---
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import socketio

# --- FastAPI Imports ---
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- NeuroPepper Routers ---
from backend.routers import (
    speech_router,
    orchestrator_router,
    text_router,
    image_router,
    video_router,
    admin_router,
    rag_router,
    models_router,
    vision_router,
    benchmark_router,
    presentation_router,
    pepper_router,
    memory_router,
    evaluation_router,
    generative_ai_router,
)

# --- NeuroPepper Core Services ---
from backend.core.performance_monitor import performance_monitor

# --- Load Environment Variables ---
load_dotenv()

# ============================================
# APPLICATION INITIALIZATION
# ============================================

# Create the main FastAPI application
app = FastAPI(
    title="NeuroPepper - Intelligent Robot AI Platform + LLM Evaluation Suite",
    description="Transform Pepper robot into a brilliant AI assistant. Compare LLMs on conversation and mental-state detection.",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Socket.IO server for real-time features
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")

# Wrap FastAPI with Socket.IO (this is what Uvicorn runs)
socket_app = socketio.ASGIApp(sio, app)

# ============================================
# STATIC FILES AND TEMPLATES
# ============================================

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"

# Create necessary directories
(DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "vectorstores").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "memory").mkdir(parents=True, exist_ok=True)

# Mount static files (only if the directories exist)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")

if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(FRONTEND_DIR))

# ============================================
# API ROUTER REGISTRATION
# ============================================

# Core AI Services
app.include_router(text_router, prefix="/text", tags=["Text Generation"])
app.include_router(vision_router, prefix="/vision", tags=["Vision Analysis"])
app.include_router(image_router, prefix="/image", tags=["Image Generation"])
app.include_router(video_router, prefix="/video", tags=["Video Generation"])

# RAG and Memory Systems
app.include_router(rag_router, prefix="/rag", tags=["RAG - Document Q&A"])
app.include_router(memory_router, prefix="/memory", tags=["Memory System"])

# Pepper Robot Control
app.include_router(pepper_router, prefix="/pepper", tags=["Pepper Robot"])

# Admin and Utilities
app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])
app.include_router(benchmark_router, prefix="/benchmark", tags=["Benchmark"])
app.include_router(evaluation_router, prefix="/eval", tags=["LLM Evaluation"])
app.include_router(presentation_router, prefix="/presentations", tags=["Presentations"])
app.include_router(speech_router, prefix="/speech", tags=["Speech"])
app.include_router(orchestrator_router, prefix="/orchestrator", tags=["Orchestrator"])
app.include_router(generative_ai_router, prefix="/generate", tags=["Generative AI (Stability/Runway/ComfyUI)"])

# ============================================
# HTML PAGE ROUTES
# ============================================

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    """Prevent 404 errors for favicon requests."""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/pepper_control", response_class=HTMLResponse)
async def serve_pepper_control(request: Request):
    """Serve the Pepper robot control panel."""
    return templates.TemplateResponse("pepper_control.html", {"request": request})


@app.get("/rag_studio", response_class=HTMLResponse)
async def serve_rag_studio(request: Request):
    """Serve the RAG document processing interface."""
    return templates.TemplateResponse("rag_studio.html", {"request": request})


@app.get("/memory_explorer", response_class=HTMLResponse)
async def serve_memory_explorer(request: Request):
    """Serve the memory system explorer."""
    return templates.TemplateResponse("memory_explorer.html", {"request": request})


@app.get("/{page_name}", response_class=HTMLResponse)
async def serve_page(request: Request, page_name: str):
    """
    Dynamic page router - serves HTML pages by name.
    Also provides model data for pages that need it (like benchmark).
    """
    template_path = FRONTEND_DIR / f"{page_name}.html"
    
    if template_path.exists():
        # Fetch available models for pages that need them
        try:
            from backend.routers.models_router import get_all_available_models
            all_models_response = await get_all_available_models()
            model_data = json.loads(all_models_response.body.decode())
            
            # Flatten all models into a single list
            model_list = []
            for category_models in model_data.values():
                if isinstance(category_models, list):
                    model_list.extend(category_models)
            
            context = {
                "request": request, 
                "models": sorted(list(set(model_list)))
            }
        except Exception:
            context = {"request": request, "models": []}
        
        return templates.TemplateResponse(f"{page_name}.html", context)
    
    # Return 404 page if not found
    return templates.TemplateResponse(
        "404.html", 
        {"request": request}, 
        status_code=404
    )

# ============================================
# SOCKET.IO EVENT HANDLERS
# ============================================

@sio.event
async def connect(sid, environ):
    """Handle new WebSocket connections."""
    print(f"✅ Client connected: {sid}")


@sio.event
async def disconnect(sid):
    """Handle WebSocket disconnections."""
    print(f"❌ Client disconnected: {sid}")


@sio.event
async def share_query(sid, data):
    """Broadcast queries to collaborative chat rooms."""
    room = data.get('room', 'general')
    await sio.emit('query_shared', data, room=room)


@sio.event
async def pepper_command(sid, data):
    """Handle real-time Pepper robot commands via WebSocket."""
    command = data.get('command')
    params = data.get('params', {})
    
    # Broadcast command status to all connected clients
    await sio.emit('pepper_status', {
        'status': 'executing',
        'command': command,
        'params': params
    })


@sio.event
async def join_room(sid, room):
    """Join a collaborative chat room."""
    sio.enter_room(sid, room)
    await sio.emit('room_joined', {'room': room}, room=sid)


@sio.event
async def leave_room(sid, room):
    """Leave a collaborative chat room."""
    sio.leave_room(sid, room)
    await sio.emit('room_left', {'room': room}, room=sid)

# ============================================
# APPLICATION LIFECYCLE
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run when the server starts."""
    print("=" * 50)
    print("🧠 NeuroPepper - Intelligent Robot AI Platform")
    print("=" * 50)
    print(f"📁 Frontend: {FRONTEND_DIR}")
    print(f"📁 Data: {DATA_DIR}")
    print("=" * 50)
    
    # Start performance monitoring
    performance_monitor.start()
    
    # Log available models
    print("🔍 Checking available models...")
    try:
        from backend.services import ollama_service
        models = await ollama_service.list_models()
        if models:
            print(f"✅ Found {len(models)} Ollama models")
        else:
            print("⚠️  No Ollama models found. Run: ollama pull qwen2.5:7b")
    except Exception as e:
        print(f"⚠️  Could not connect to Ollama: {e}")
    
    print("=" * 50)
    print("🚀 Server ready at http://localhost:5000")
    print("📚 API docs at http://localhost:5000/api/docs")
    print("=" * 50)


@app.on_event("shutdown")
def shutdown_event():
    """Run when the server shuts down."""
    print("\n👋 Shutting down NeuroPepper...")
    performance_monitor.stop()


# ============================================
# HEALTH CHECK ENDPOINT
# ============================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "NeuroPepper",
        "version": "3.0.0"
    }


# ============================================
# HOW TO RUN
# ============================================
# 
# Method 1: Using the launcher script
#   python run_server.py
#
# Method 2: Direct uvicorn command
#   uvicorn backend.app:socket_app --reload --host 0.0.0.0 --port 5000
#
# Method 3: Production (multiple workers)
#   uvicorn backend.app:socket_app --workers 4 --host 0.0.0.0 --port 5000
#
# ============================================
