# backend/services/ollama_service.py
# ============================================
# Ollama Service - Local LLM Integration
# ============================================
# Handles communication with the Ollama server for:
# - Text generation (streaming and non-streaming)
# - Vision/multimodal models (LLaVA, etc.)
# - Embedding generation for RAG
# - Model management
# ============================================

import os
import httpx
import json
import base64
from typing import List, Optional, AsyncGenerator, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Get Ollama server URL from environment
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ============================================
# MODEL MANAGEMENT
# ============================================

async def list_models() -> List[str]:
    """
    Fetch all available models from the Ollama server.
    
    Returns:
        List of model IDs in format "ollama/model_name"
    
    Example:
        >>> models = await list_models()
        >>> print(models)
        ['ollama/qwen2.5:7b', 'ollama/llava:latest', ...]
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            models_data = response.json().get('models', [])
            return [f"ollama/{model['name']}" for model in models_data]
        except httpx.RequestError as e:
            print(f"⚠️ Could not connect to Ollama at {OLLAMA_BASE_URL}: {e}")
            return []
        except Exception as e:
            print(f"⚠️ Error listing models: {e}")
            return []


async def is_model_available(model: str) -> bool:
    """Check if a specific model is available."""
    models = await list_models()
    model_name = model if model.startswith("ollama/") else f"ollama/{model}"
    return model_name in models


async def get_model_info(model: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific model."""
    actual_model = model.split('/')[-1] if '/' in model else model
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/show",
                json={"name": actual_model}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️ Error getting model info: {e}")
            return None

# ============================================
# TEXT GENERATION
# ============================================

async def generate_text(
    prompt: str, 
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> str:
    """
    Generate text response (non-streaming).
    
    Args:
        prompt: User's input text
        model: Model ID (e.g., "ollama/qwen2.5:7b" or "qwen2.5:7b")
        system_prompt: Optional system instructions
        temperature: Creativity (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum response length
    
    Returns:
        Generated text response
    """
    full_response = ""
    async for chunk in generate_text_stream(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    ):
        full_response += chunk
    return full_response


async def generate_text_stream(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    images: Optional[List[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> AsyncGenerator[str, None]:
    """
    Generate text response with streaming.
    
    This is the preferred method for real-time responses as it
    yields tokens as they're generated.
    
    Args:
        prompt: User's input text
        model: Model ID
        system_prompt: Optional system instructions
        images: Optional list of base64-encoded images (for vision models)
        temperature: Creativity level
        max_tokens: Maximum response length
    
    Yields:
        Individual tokens as they're generated
    
    Example:
        async for token in generate_text_stream("Hello", "qwen2.5:7b"):
            print(token, end="", flush=True)
    """
    # Extract actual model name (remove "ollama/" prefix if present)
    actual_model = model.split('/')[-1] if '/' in model else model
    
    # Build the request payload
    payload = {
        "model": actual_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    
    # Add system prompt if provided
    if system_prompt:
        payload["system"] = system_prompt
    
    # Add images for vision models
    if images:
        payload["images"] = images
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get('response', '')
                            if token:
                                yield token
                            
                            # Check if generation is complete
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPStatusError as e:
            yield f"\n\n⚠️ Error from Ollama: {e.response.text}"
        except httpx.RequestError as e:
            yield f"\n\n⚠️ Connection error: Could not reach Ollama at {OLLAMA_BASE_URL}"
        except Exception as e:
            yield f"\n\n⚠️ Unexpected error: {str(e)}"


async def chat(
    messages: List[Dict[str, str]],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> str:
    """
    Chat-style conversation with message history.
    
    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}
        model: Model ID
        system_prompt: System instructions
        temperature: Creativity level
        max_tokens: Maximum response length
    
    Returns:
        Assistant's response
    """
    actual_model = model.split('/')[-1] if '/' in model else model
    
    payload = {
        "model": actual_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    
    if system_prompt:
        payload["messages"].insert(0, {"role": "system", "content": system_prompt})
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            raise ConnectionError(f"Ollama chat error: {str(e)}")

# ============================================
# VISION / MULTIMODAL
# ============================================

async def generate_with_image(
    prompt: str,
    image_bytes: bytes,
    model: str = "llava:latest"
) -> str:
    """
    Generate response from image + text prompt using a vision model.
    
    Args:
        prompt: Question or instruction about the image
        image_bytes: Raw image bytes (JPEG, PNG, etc.)
        model: Vision model ID (e.g., "llava:latest", "bakllava:latest")
    
    Returns:
        Model's analysis of the image
    
    Example:
        with open("photo.jpg", "rb") as f:
            image_data = f.read()
        response = await generate_with_image(
            "What do you see in this image?",
            image_data,
            "llava:latest"
        )
    """
    actual_model = model.split('/')[-1] if '/' in model else model
    
    # Encode image to base64
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    payload = {
        "model": actual_model,
        "prompt": prompt,
        "stream": False,
        "images": [encoded_image]
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload
            )
            response.raise_for_status()
            return response.json().get('response', '')
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"Ollama Vision Error: {e.response.text}")
        except Exception as e:
            raise ConnectionError(f"Vision generation failed: {str(e)}")


async def analyze_image_stream(
    prompt: str,
    image_bytes: bytes,
    model: str = "llava:latest"
) -> AsyncGenerator[str, None]:
    """
    Stream analysis of an image (for real-time UI updates).
    """
    actual_model = model.split('/')[-1] if '/' in model else model
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    async for token in generate_text_stream(
        prompt=prompt,
        model=actual_model,
        images=[encoded_image]
    ):
        yield token

# ============================================
# EMBEDDINGS (for RAG)
# ============================================

async def generate_embedding(
    text: str,
    model: str = "nomic-embed-text"
) -> List[float]:
    """
    Generate embedding vector for text.
    
    Args:
        text: Text to embed
        model: Embedding model (default: nomic-embed-text)
    
    Returns:
        List of floats representing the embedding vector
    """
    actual_model = model.split('/')[-1] if '/' in model else model
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": actual_model, "prompt": text}
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            raise ConnectionError(f"Embedding generation failed: {str(e)}")


async def generate_embeddings_batch(
    texts: List[str],
    model: str = "nomic-embed-text"
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    """
    embeddings = []
    for text in texts:
        embedding = await generate_embedding(text, model)
        embeddings.append(embedding)
    return embeddings

# ============================================
# UTILITY FUNCTIONS
# ============================================

async def check_server_health() -> Dict[str, Any]:
    """
    Check if Ollama server is running and healthy.
    
    Returns:
        Dict with status information
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            return {
                "status": "healthy",
                "url": OLLAMA_BASE_URL,
                "models_count": len(models)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "url": OLLAMA_BASE_URL,
                "error": str(e)
            }


def get_recommended_models() -> Dict[str, str]:
    """
    Get recommended models for different tasks.
    """
    return {
        "text_generation": "qwen2.5:7b",
        "text_fast": "qwen2.5:3b",
        "vision": "llava:latest",
        "embedding": "nomic-embed-text",
        "code": "codellama:7b",
    }
