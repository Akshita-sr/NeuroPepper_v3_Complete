# backend/routers/benchmark_router.py
# ============================================
# Model Benchmark Router
# ============================================

import time
import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from backend.services import ollama_service

router = APIRouter()


class BenchmarkRequest(BaseModel):
    models: List[str]
    prompt: str


@router.post("/run")
async def run_benchmark(request: BenchmarkRequest):
    """
    Run benchmarks on multiple models with the same prompt.
    
    Measures latency, response length, and tokens/second.
    """
    results = {}
    
    for model_id in request.models:
        provider, model_name = model_id.split('/', 1)
        
        if provider != "ollama":
            results[model_id] = {
                "status": "skipped",
                "error": "Only Ollama models supported for benchmarking"
            }
            continue
        
        try:
            start_time = time.time()
            
            response = await ollama_service.generate_text(
                prompt=request.prompt,
                model=model_name
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Estimate tokens (rough: ~4 chars per token)
            response_length = len(response)
            estimated_tokens = response_length / 4
            tokens_per_sec = estimated_tokens / latency if latency > 0 else 0
            
            results[model_id] = {
                "status": "success",
                "latency": f"{latency:.2f}s",
                "response_length": response_length,
                "tokens_per_sec": f"{tokens_per_sec:.1f}"
            }
            
        except Exception as e:
            results[model_id] = {
                "status": "failed",
                "error": str(e)
            }
    
    return results
