# backend/routers/evaluation_router.py
# ============================================
# Evaluation Router - API for LLM Evaluation
# ============================================
# Provides HTTP endpoints for running evaluations
# from the web interface or external scripts.
# ============================================

import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict

from backend.services.multi_llm_service import (
    call_model, call_models_parallel, get_available_models,
    discover_ollama_models, check_model_availability,
)
from backend.core.evaluation_engine import (
    MENTAL_STATE_SCENARIOS, CONVERSATION_PROMPTS,
    build_mental_state_prompt, build_conversation_system_prompt,
    parse_mental_state_response, score_mental_state,
    auto_score_conversation,
    aggregate_mental_state_results, aggregate_conversation_results,
    MentalStateResult, ConversationResult,
)
from backend.core.user_profile_db import (
    get_user, list_users, create_user, save_eval_session, get_eval_summary,
)

router = APIRouter()


# ============================================
# REQUEST / RESPONSE MODELS
# ============================================

class MultiModelRequest(BaseModel):
    prompt: str
    model_ids: List[str]
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048


class EvalRequest(BaseModel):
    model_ids: List[str]
    user_id: Optional[str] = None


class UserProfileRequest(BaseModel):
    user_id: str
    name: str
    age_group: str = "adult"
    language: str = "en"
    preferences: Dict = {}
    context_notes: str = ""


class UserRatingRequest(BaseModel):
    session_id: str
    user_id: str
    model_id: str
    task_type: str
    prompt: str
    response: str
    user_rating: int
    user_feedback: str = ""
    latency: float = 0.0


# ============================================
# MULTI-MODEL ENDPOINTS
# ============================================

@router.get("/models")
async def list_all_models():
    """List all configured + installed models."""
    configured = [
        {"model_id": m.model_id, "provider": m.provider, "display_name": m.display_name}
        for m in get_available_models()
    ]
    installed_ollama = await discover_ollama_models()
    return {
        "configured": configured,
        "installed_ollama": installed_ollama,
    }


@router.post("/models/check")
async def check_models(model_ids: List[str]):
    """Check availability of specified models."""
    tasks = [check_model_availability(mid) for mid in model_ids]
    results = await asyncio.gather(*tasks)
    return {"results": results}


@router.post("/compare")
async def compare_models(request: MultiModelRequest):
    """Send same prompt to multiple models and compare responses."""
    results = await call_models_parallel(
        request.model_ids, request.prompt,
        request.system_prompt, request.temperature, request.max_tokens,
    )
    return {
        "prompt": request.prompt,
        "results": [r.to_dict() for r in results],
    }


# ============================================
# MENTAL STATE EVALUATION
# ============================================

@router.post("/eval/mental-state")
async def run_mental_state_eval(request: EvalRequest):
    """Run mental-state detection evaluation on specified models."""
    user_context = ""
    if request.user_id:
        user = get_user(request.user_id)
        if user:
            user_context = user.to_system_context()

    all_results = {}
    for mid in request.model_ids:
        results = []
        for scenario in MENTAL_STATE_SCENARIOS:
            prompt = build_mental_state_prompt(scenario, user_context)
            resp = await call_model(mid, prompt, max_tokens=200)

            if resp.success:
                parsed = parse_mental_state_response(resp.response_text)
                scoring = score_mental_state(
                    parsed["detected_state"], scenario["correct_state"]
                )
                results.append(MentalStateResult(
                    scenario_id=scenario["id"], model_id=mid,
                    user_message=scenario["user_message"],
                    correct_state=scenario["correct_state"],
                    detected_state=parsed["detected_state"],
                    confidence=parsed["confidence"],
                    is_correct=scoring["correct"],
                    score=scoring["score"],
                    latency=resp.latency_seconds,
                    raw_response=resp.response_text[:300],
                ))

        all_results[mid] = {
            "results": [r.to_dict() for r in results],
            "summary": aggregate_mental_state_results(results),
        }

    return all_results


# ============================================
# CONVERSATION EVALUATION
# ============================================

@router.post("/eval/conversation")
async def run_conversation_eval(request: EvalRequest):
    """Run conversation quality evaluation on specified models."""
    system_prompt = build_conversation_system_prompt()
    if request.user_id:
        user = get_user(request.user_id)
        if user:
            system_prompt = build_conversation_system_prompt(user.to_system_context())

    all_results = {}
    for mid in request.model_ids:
        results = []
        for conv in CONVERSATION_PROMPTS:
            resp = await call_model(mid, conv["prompt"], system_prompt=system_prompt)

            if resp.success:
                auto_scores = auto_score_conversation(
                    resp.response_text, conv.get("quality_criteria", [])
                )
                results.append(ConversationResult(
                    prompt_id=conv["id"], model_id=mid,
                    category=conv["category"], prompt=conv["prompt"],
                    response=resp.response_text, auto_scores=auto_scores,
                    latency=resp.latency_seconds,
                ))

        all_results[mid] = {
            "results": [r.to_dict() for r in results],
            "summary": aggregate_conversation_results(results),
        }

    return all_results


# ============================================
# USER PROFILE ENDPOINTS
# ============================================

@router.get("/users")
async def get_users():
    """List all user profiles."""
    users = list_users()
    return [
        {"user_id": u.user_id, "name": u.name, "age_group": u.age_group,
         "language": u.language}
        for u in users
    ]


@router.post("/users")
async def create_user_profile(request: UserProfileRequest):
    """Create or update a user profile."""
    user = create_user(
        request.user_id, request.name, request.age_group,
        request.language, request.preferences, request.context_notes,
    )
    return {"status": "created", "user_id": user.user_id}


# ============================================
# USER RATING / FEEDBACK
# ============================================

@router.post("/eval/rate")
async def submit_rating(request: UserRatingRequest):
    """Submit a user rating for a model response."""
    save_eval_session(
        request.session_id, request.user_id, request.model_id,
        request.task_type, request.prompt, request.response,
        request.user_rating, request.user_feedback,
        latency=request.latency,
    )
    return {"status": "saved", "session_id": request.session_id}


@router.get("/eval/summary")
async def evaluation_summary():
    """Get aggregate evaluation summary across all models and tasks."""
    return get_eval_summary()
