# backend/routers/rag_router.py
# ============================================
# RAG (Retrieval Augmented Generation) Router
# ============================================

import os
import base64
import json
import aiofiles
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

from backend.core.rag_manager import rag_manager
from backend.services import ollama_service

router = APIRouter()

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class RAGQueryRequest(BaseModel):
    prompt: str
    vectorstore_path: str
    model: str = "ollama/qwen2.5:7b"


@router.websocket("/upload")
async def handle_rag_upload(websocket: WebSocket):
    """
    WebSocket endpoint for document upload with progress updates.
    
    Expects JSON: {filename: str, content: base64_string}
    """
    await websocket.accept()
    
    try:
        data = await websocket.receive_json()
        filename = data.get('filename', 'document.pdf')
        file_content_base64 = data.get('content')
        
        if not file_content_base64:
            raise ValueError("File content is missing")
        
        # Decode and save file
        file_bytes = base64.b64decode(file_content_base64)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(file_bytes)
        
        # Progress callback
        async def progress_callback(msg: str):
            await websocket.send_json({"type": "update", "message": msg})
        
        # Process document
        vectorstore_path = await rag_manager.process_and_store_document(
            filepath, 
            progress_callback=progress_callback
        )
        
        await websocket.send_json({
            "type": "finished",
            "filename": filename,
            "path": vectorstore_path
        })
        
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
    finally:
        await websocket.close()


@router.post("/query")
async def query_rag_document(request: RAGQueryRequest):
    """
    Query a document using RAG with streaming response.
    """
    def get_query_function(model_id: str):
        provider, model_name = model_id.split('/', 1)
        
        if provider == "ollama":
            async def ollama_query(prompt_text: str):
                async for token in ollama_service.generate_text_stream(prompt_text, model_name):
                    yield token
            return ollama_query
        
        raise HTTPException(400, f"Unsupported provider: {provider}")
    
    query_func = get_query_function(request.model)
    
    async def stream_wrapper():
        try:
            async for token in rag_manager.query_with_rag_stream(
                question=request.prompt,
                vectorstore_path=request.vectorstore_path,
                query_function=query_func
            ):
                yield f"data: {json.dumps({'type': 'content', 'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")


@router.get("/vectorstores")
async def list_vectorstores():
    """List all available vectorstores."""
    return {"vectorstores": rag_manager.list_vectorstores()}


@router.delete("/vectorstore")
async def delete_vectorstore(path: str):
    """Delete a vectorstore."""
    success = rag_manager.delete_vectorstore(path)
    if success:
        return {"success": True}
    raise HTTPException(404, "Vectorstore not found")
