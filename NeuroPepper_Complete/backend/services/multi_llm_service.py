# backend/services/multi_llm_service.py
# ============================================
# Multi-LLM Service - Unified Interface
# ============================================
# Connects to ALL supported LLMs through one interface:
# - Ollama (any local model: Qwen, LLaMA, Mistral, etc.)
# - DeepSeek Local (your lab server running DeepSeek)
# - DeepSeek Cloud (api.deepseek.com)
# - OpenAI GPT-4 / GPT-4o
# - Anthropic Claude
# ============================================

import os
import time
import httpx
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEEPSEEK_LOCAL_URL = os.getenv("DEEPSEEK_LOCAL_URL", "http://localhost:8000/v1")
DEEPSEEK_CLOUD_URL = os.getenv("DEEPSEEK_CLOUD_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Timeout for API calls (seconds)
API_TIMEOUT = 120.0


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class LLMResponse:
    """Holds everything about one LLM response."""
    model_id: str           # e.g. "ollama/qwen2.5:7b"
    provider: str           # e.g. "ollama", "deepseek_local", "openai"
    response_text: str      # The actual generated text
    latency_seconds: float  # How long it took (seconds)
    prompt_tokens: int      # Input tokens (estimated if not provided)
    completion_tokens: int  # Output tokens (estimated if not provided)
    success: bool           # Did it work?
    error: str = ""         # Error message if it failed
    raw_response: Dict = field(default_factory=dict)  # Full API response

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelConfig:
    """Configuration for one LLM."""
    model_id: str           # Unique identifier
    provider: str           # Which provider
    model_name: str         # Name the API expects
    base_url: str           # API endpoint
    api_key: str = ""       # API key (if needed)
    display_name: str = ""  # Human-readable name


# ============================================
# PROVIDER REGISTRY
# ============================================

def get_available_models() -> List[ModelConfig]:
    """
    Returns a list of all models that are configured and potentially available.
    Each model has a provider, API endpoint, and key (if needed).
    """
    models = []

    # --- Ollama models (always available if Ollama is running) ---
    # Common models people use. The CLI also auto-discovers what's installed.
    ollama_defaults = [
        "qwen2.5:7b", "qwen2.5:3b", "llama3.1:8b", "mistral:7b",
        "deepseek-r1:7b", "deepseek-r1:14b", "phi3:mini",
        "gemma2:9b", "codellama:7b",
    ]
    for m in ollama_defaults:
        models.append(ModelConfig(
            model_id=f"ollama/{m}",
            provider="ollama",
            model_name=m,
            base_url=OLLAMA_BASE_URL,
            display_name=f"Ollama {m}",
        ))

    # --- DeepSeek Local (your lab server) ---
    models.append(ModelConfig(
        model_id="deepseek_local/deepseek-chat",
        provider="deepseek_local",
        model_name="deepseek-chat",
        base_url=DEEPSEEK_LOCAL_URL,
        display_name="DeepSeek (Local Server)",
    ))

    # --- DeepSeek Cloud ---
    if DEEPSEEK_API_KEY:
        models.append(ModelConfig(
            model_id="deepseek_cloud/deepseek-chat",
            provider="deepseek_cloud",
            model_name="deepseek-chat",
            base_url=DEEPSEEK_CLOUD_URL,
            api_key=DEEPSEEK_API_KEY,
            display_name="DeepSeek (Cloud)",
        ))
        models.append(ModelConfig(
            model_id="deepseek_cloud/deepseek-reasoner",
            provider="deepseek_cloud",
            model_name="deepseek-reasoner",
            base_url=DEEPSEEK_CLOUD_URL,
            api_key=DEEPSEEK_API_KEY,
            display_name="DeepSeek Reasoner (Cloud)",
        ))

    # --- OpenAI GPT-4 ---
    if OPENAI_API_KEY:
        for m_name, m_display in [
            ("gpt-4o", "GPT-4o"),
            ("gpt-4-turbo", "GPT-4 Turbo"),
            ("gpt-4", "GPT-4"),
            ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
        ]:
            models.append(ModelConfig(
                model_id=f"openai/{m_name}",
                provider="openai",
                model_name=m_name,
                base_url="https://api.openai.com/v1",
                api_key=OPENAI_API_KEY,
                display_name=m_display,
            ))

    # --- Anthropic Claude ---
    if ANTHROPIC_API_KEY:
        for m_name, m_display in [
            ("claude-sonnet-4-20250514", "Claude Sonnet 4"),
            ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ]:
            models.append(ModelConfig(
                model_id=f"claude/{m_name}",
                provider="claude",
                model_name=m_name,
                base_url="https://api.anthropic.com/v1",
                api_key=ANTHROPIC_API_KEY,
                display_name=m_display,
            ))

    return models


def find_model(model_id: str) -> Optional[ModelConfig]:
    """Find a model by its ID (e.g., 'openai/gpt-4o')."""
    for m in get_available_models():
        if m.model_id == model_id:
            return m
    return None


# ============================================
# PROVIDER-SPECIFIC CALLERS
# ============================================

async def _call_ollama(config: ModelConfig, prompt: str,
                       system_prompt: str = "", temperature: float = 0.7,
                       max_tokens: int = 2048) -> LLMResponse:
    """Call an Ollama model using /api/generate."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            payload = {
                "model": config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if system_prompt:
                payload["system"] = system_prompt

            resp = await client.post(
                f"{config.base_url}/api/generate", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

            latency = time.time() - start
            text = data.get("response", "")
            p_tokens = data.get("prompt_eval_count", _estimate_tokens(prompt))
            c_tokens = data.get("eval_count", _estimate_tokens(text))

            return LLMResponse(
                model_id=config.model_id, provider="ollama",
                response_text=text, latency_seconds=latency,
                prompt_tokens=p_tokens, completion_tokens=c_tokens,
                success=True, raw_response=data,
            )
    except Exception as e:
        return LLMResponse(
            model_id=config.model_id, provider="ollama",
            response_text="", latency_seconds=time.time() - start,
            prompt_tokens=0, completion_tokens=0,
            success=False, error=str(e),
        )


async def _call_openai_compatible(config: ModelConfig, prompt: str,
                                   system_prompt: str = "",
                                   temperature: float = 0.7,
                                   max_tokens: int = 2048) -> LLMResponse:
    """
    Calls any OpenAI-compatible API (OpenAI, DeepSeek local/cloud).
    They all use the same /chat/completions format.
    """
    start = time.time()
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            resp = await client.post(
                f"{config.base_url}/chat/completions",
                json=payload, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        latency = time.time() - start
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            model_id=config.model_id, provider=config.provider,
            response_text=text, latency_seconds=latency,
            prompt_tokens=usage.get("prompt_tokens", _estimate_tokens(prompt)),
            completion_tokens=usage.get("completion_tokens", _estimate_tokens(text)),
            success=True, raw_response=data,
        )
    except Exception as e:
        return LLMResponse(
            model_id=config.model_id, provider=config.provider,
            response_text="", latency_seconds=time.time() - start,
            prompt_tokens=0, completion_tokens=0,
            success=False, error=str(e),
        )


async def _call_claude(config: ModelConfig, prompt: str,
                       system_prompt: str = "", temperature: float = 0.7,
                       max_tokens: int = 2048) -> LLMResponse:
    """Call Anthropic Claude API (uses a different format than OpenAI)."""
    start = time.time()
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": config.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            resp = await client.post(
                f"{config.base_url}/messages",
                json=payload, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        latency = time.time() - start
        # Claude returns content as a list of blocks
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = data.get("usage", {})
        return LLMResponse(
            model_id=config.model_id, provider="claude",
            response_text=text, latency_seconds=latency,
            prompt_tokens=usage.get("input_tokens", _estimate_tokens(prompt)),
            completion_tokens=usage.get("output_tokens", _estimate_tokens(text)),
            success=True, raw_response=data,
        )
    except Exception as e:
        return LLMResponse(
            model_id=config.model_id, provider="claude",
            response_text="", latency_seconds=time.time() - start,
            prompt_tokens=0, completion_tokens=0,
            success=False, error=str(e),
        )


# ============================================
# UNIFIED CALLER
# ============================================

async def call_model(model_id: str, prompt: str,
                     system_prompt: str = "", temperature: float = 0.7,
                     max_tokens: int = 2048) -> LLMResponse:
    """
    Call ANY model by its ID. This is the main entry point.

    Args:
        model_id: e.g. "openai/gpt-4o", "ollama/qwen2.5:7b", "claude/claude-sonnet-4-20250514"
        prompt: The user message
        system_prompt: Optional system instructions
        temperature: Creativity (0.0 = focused, 1.0 = creative)
        max_tokens: Maximum response length

    Returns:
        LLMResponse with text, latency, token counts, and success status
    """
    config = find_model(model_id)

    # If model not in registry, try to parse it as ollama model
    if config is None:
        if "/" in model_id:
            provider, model_name = model_id.split("/", 1)
            config = ModelConfig(
                model_id=model_id, provider=provider,
                model_name=model_name,
                base_url=OLLAMA_BASE_URL if provider == "ollama" else "",
            )
        else:
            return LLMResponse(
                model_id=model_id, provider="unknown",
                response_text="", latency_seconds=0,
                prompt_tokens=0, completion_tokens=0,
                success=False, error=f"Unknown model: {model_id}",
            )

    # Route to the correct provider
    if config.provider == "ollama":
        return await _call_ollama(config, prompt, system_prompt, temperature, max_tokens)
    elif config.provider in ("openai", "deepseek_local", "deepseek_cloud"):
        return await _call_openai_compatible(config, prompt, system_prompt, temperature, max_tokens)
    elif config.provider == "claude":
        return await _call_claude(config, prompt, system_prompt, temperature, max_tokens)
    else:
        return LLMResponse(
            model_id=model_id, provider=config.provider,
            response_text="", latency_seconds=0,
            prompt_tokens=0, completion_tokens=0,
            success=False, error=f"Unsupported provider: {config.provider}",
        )


async def call_models_parallel(model_ids: List[str], prompt: str,
                                system_prompt: str = "",
                                temperature: float = 0.7,
                                max_tokens: int = 2048) -> List[LLMResponse]:
    """
    Call multiple models at the same time (in parallel).
    Returns a list of LLMResponse, one per model.
    """
    import asyncio
    tasks = [
        call_model(mid, prompt, system_prompt, temperature, max_tokens)
        for mid in model_ids
    ]
    return await asyncio.gather(*tasks)


# ============================================
# UTILITIES
# ============================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


async def check_model_availability(model_id: str) -> Dict[str, Any]:
    """Check if a model is reachable and responding."""
    try:
        result = await call_model(model_id, "Say hello.", max_tokens=20)
        return {
            "model_id": model_id,
            "available": result.success,
            "latency": f"{result.latency_seconds:.2f}s",
            "error": result.error if not result.success else None,
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "available": False,
            "error": str(e),
        }


async def discover_ollama_models() -> List[str]:
    """Ask Ollama what models are actually installed."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [f"ollama/{m['name']}" for m in models]
    except Exception:
        return []
