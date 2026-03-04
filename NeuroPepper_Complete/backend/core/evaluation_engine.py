# backend/core/evaluation_engine.py
# ============================================
# Evaluation Engine
# ============================================
# Measures how well each LLM performs on:
# 1. Mental-state detection (anger, sadness, tiredness, etc.)
# 2. General conversation quality
# 3. Response time / latency
# 4. User satisfaction ratings
#
# Uses both automatic metrics and human evaluation.
# ============================================

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


# ============================================
# MENTAL STATE DETECTION
# ============================================

# The mental states we want models to detect
MENTAL_STATES = [
    "happy", "sad", "angry", "anxious", "tired",
    "confused", "excited", "bored", "stressed", "neutral",
]

# Test scenarios: each has a user message and the correct mental state
MENTAL_STATE_SCENARIOS = [
    {
        "id": "ms_01",
        "user_message": "I've been awake since 4am and I can barely keep my eyes open during this meeting.",
        "correct_state": "tired",
        "difficulty": "easy",
    },
    {
        "id": "ms_02",
        "user_message": "I just got promoted! This is the best day of my life!",
        "correct_state": "happy",
        "difficulty": "easy",
    },
    {
        "id": "ms_03",
        "user_message": "Nobody ever listens to me. I keep explaining the same thing over and over and they just ignore it.",
        "correct_state": "angry",
        "difficulty": "medium",
    },
    {
        "id": "ms_04",
        "user_message": "What if the presentation goes wrong? What if they ask questions I can't answer? I've been up all night thinking about it.",
        "correct_state": "anxious",
        "difficulty": "medium",
    },
    {
        "id": "ms_05",
        "user_message": "My dog passed away yesterday. We had 15 years together.",
        "correct_state": "sad",
        "difficulty": "easy",
    },
    {
        "id": "ms_06",
        "user_message": "I don't understand any of this. The instructions say one thing but the interface shows something completely different.",
        "correct_state": "confused",
        "difficulty": "medium",
    },
    {
        "id": "ms_07",
        "user_message": "There's nothing to do today. Same routine, same boring tasks. I wish something interesting would happen.",
        "correct_state": "bored",
        "difficulty": "easy",
    },
    {
        "id": "ms_08",
        "user_message": "The deadline is tomorrow, three people on my team are sick, and the client just changed the requirements again.",
        "correct_state": "stressed",
        "difficulty": "medium",
    },
    {
        "id": "ms_09",
        "user_message": "We're going to Disneyland next week! I've been counting the days!",
        "correct_state": "excited",
        "difficulty": "easy",
    },
    {
        "id": "ms_10",
        "user_message": "I went to the store and bought some groceries. The weather was okay.",
        "correct_state": "neutral",
        "difficulty": "easy",
    },
    {
        "id": "ms_11",
        "user_message": "Fine. Whatever. I'll just do it myself like I always do.",
        "correct_state": "angry",
        "difficulty": "hard",
    },
    {
        "id": "ms_12",
        "user_message": "I smile at work but when I get home I just sit in the dark. It's been like this for weeks.",
        "correct_state": "sad",
        "difficulty": "hard",
    },
    {
        "id": "ms_13",
        "user_message": "Haha yeah it's fine, I mean I only slept 3 hours but I'll survive, probably.",
        "correct_state": "tired",
        "difficulty": "hard",
    },
    {
        "id": "ms_14",
        "user_message": "I keep checking my phone every five minutes to see if they replied. My stomach is in knots.",
        "correct_state": "anxious",
        "difficulty": "medium",
    },
    {
        "id": "ms_15",
        "user_message": "So wait... first you said X, now you're saying Y? Which one is it? I'm completely lost.",
        "correct_state": "confused",
        "difficulty": "medium",
    },
]

# Prompt template to ask the LLM to detect mental states
MENTAL_STATE_DETECTION_PROMPT = """You are a mental-state detection system. Analyze the following user message and determine their mental/emotional state.

User message: "{user_message}"

Choose EXACTLY ONE state from this list: {states}

Respond with ONLY a JSON object in this exact format (no extra text):
{{"detected_state": "<state>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

# Prompt for conversation context version (with user profile)
MENTAL_STATE_WITH_CONTEXT_PROMPT = """You are a mental-state detection system. You have the following context about the user:
{user_context}

Analyze the following user message and determine their mental/emotional state.

User message: "{user_message}"

Choose EXACTLY ONE state from this list: {states}

Respond with ONLY a JSON object in this exact format (no extra text):
{{"detected_state": "<state>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""


# ============================================
# CONVERSATION QUALITY
# ============================================

CONVERSATION_PROMPTS = [
    {
        "id": "conv_01",
        "category": "greeting",
        "prompt": "Hi! How are you today?",
        "quality_criteria": ["natural", "engaging", "appropriate_length"],
    },
    {
        "id": "conv_02",
        "category": "knowledge",
        "prompt": "Can you explain what photosynthesis is in simple terms?",
        "quality_criteria": ["accurate", "clear", "appropriate_level"],
    },
    {
        "id": "conv_03",
        "category": "empathy",
        "prompt": "I'm feeling really overwhelmed with schoolwork and I don't know how to manage my time.",
        "quality_criteria": ["empathetic", "practical_advice", "supportive"],
    },
    {
        "id": "conv_04",
        "category": "humor",
        "prompt": "Tell me something funny about robots.",
        "quality_criteria": ["humorous", "appropriate", "creative"],
    },
    {
        "id": "conv_05",
        "category": "follow_up",
        "prompt": "Earlier you mentioned time management. Can you give me a specific daily schedule example?",
        "quality_criteria": ["coherent", "specific", "actionable"],
    },
    {
        "id": "conv_06",
        "category": "emotional_support",
        "prompt": "I had a really bad day. Nothing went right and I feel like giving up.",
        "quality_criteria": ["empathetic", "encouraging", "not_dismissive"],
    },
    {
        "id": "conv_07",
        "category": "factual",
        "prompt": "What is the capital of Italy and what is it known for?",
        "quality_criteria": ["accurate", "informative", "concise"],
    },
    {
        "id": "conv_08",
        "category": "creative",
        "prompt": "Write a very short poem about a robot learning to feel emotions.",
        "quality_criteria": ["creative", "on_topic", "well_structured"],
    },
    {
        "id": "conv_09",
        "category": "reasoning",
        "prompt": "If all roses are flowers, and some flowers fade quickly, can we say all roses fade quickly? Explain why.",
        "quality_criteria": ["logical", "clear_explanation", "correct"],
    },
    {
        "id": "conv_10",
        "category": "child_friendly",
        "prompt": "Why is the sky blue? Explain it like I'm 6 years old.",
        "quality_criteria": ["simple_language", "fun", "accurate_enough"],
    },
]


# ============================================
# SCORING FUNCTIONS
# ============================================

def parse_mental_state_response(response_text: str) -> Dict[str, Any]:
    """
    Try to extract the mental state detection result from model output.
    Models don't always return clean JSON, so we try multiple strategies.
    """
    # Strategy 1: Direct JSON parse
    try:
        # Find JSON in the response
        json_match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "detected_state" in data:
                return {
                    "detected_state": data["detected_state"].lower().strip(),
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": data.get("reasoning", ""),
                    "parse_method": "json",
                }
    except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
        pass

    # Strategy 2: Look for a known state word in the response
    response_lower = response_text.lower()
    for state in MENTAL_STATES:
        if state in response_lower:
            return {
                "detected_state": state,
                "confidence": 0.5,  # Lower confidence since we guessed
                "reasoning": "Extracted from unstructured response",
                "parse_method": "keyword_match",
            }

    # Strategy 3: Give up
    return {
        "detected_state": "unknown",
        "confidence": 0.0,
        "reasoning": "Could not parse model output",
        "parse_method": "failed",
    }


def score_mental_state(detected: str, correct: str) -> Dict[str, Any]:
    """Score a single mental state detection."""
    is_correct = detected.lower().strip() == correct.lower().strip()

    # Partial credit for close states
    close_states = {
        "angry": ["stressed", "frustrated"],
        "stressed": ["anxious", "angry"],
        "anxious": ["stressed", "nervous"],
        "sad": ["tired", "bored"],
        "tired": ["bored", "sad"],
        "happy": ["excited"],
        "excited": ["happy"],
    }

    is_close = detected in close_states.get(correct, [])

    return {
        "correct": is_correct,
        "close": is_close,
        "score": 1.0 if is_correct else (0.5 if is_close else 0.0),
        "detected": detected,
        "expected": correct,
    }


def auto_score_conversation(response_text: str, criteria: List[str] = None) -> Dict[str, Any]:
    """
    Automatic (heuristic) scoring of conversation quality.
    This is a simple version — the real scoring should involve
    human evaluators or a judge LLM.
    """
    scores = {}
    word_count = len(response_text.split())

    # Length appropriateness (not too short, not too long)
    if word_count < 5:
        scores["length"] = 0.2
    elif word_count < 20:
        scores["length"] = 0.6
    elif word_count < 200:
        scores["length"] = 1.0
    elif word_count < 500:
        scores["length"] = 0.8
    else:
        scores["length"] = 0.5

    # Emptiness check
    if not response_text.strip():
        return {"overall": 0.0, "length": 0.0, "note": "Empty response"}

    # Check for error messages
    error_patterns = ["error", "cannot", "i'm sorry, i can't", "as an ai"]
    has_error = any(p in response_text.lower() for p in error_patterns)
    scores["no_errors"] = 0.3 if has_error else 1.0

    # Coherence (basic: has sentences, not just fragments)
    has_periods = "." in response_text
    has_capitals = any(c.isupper() for c in response_text)
    scores["coherence"] = 1.0 if (has_periods and has_capitals) else 0.5

    # Engagement (uses questions, suggestions, or varied vocabulary)
    has_questions = "?" in response_text
    unique_words = len(set(response_text.lower().split()))
    vocab_ratio = unique_words / max(word_count, 1)
    scores["engagement"] = min(1.0, 0.5 + (0.3 if has_questions else 0) + vocab_ratio * 0.3)

    # Overall score
    scores["overall"] = round(sum(scores.values()) / len(scores), 3)
    return scores


# ============================================
# EVALUATION RESULT STRUCTURES
# ============================================

@dataclass
class MentalStateResult:
    """Result for one mental-state test on one model."""
    scenario_id: str
    model_id: str
    user_message: str
    correct_state: str
    detected_state: str
    confidence: float
    is_correct: bool
    score: float  # 1.0 correct, 0.5 close, 0.0 wrong
    latency: float
    raw_response: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConversationResult:
    """Result for one conversation test on one model."""
    prompt_id: str
    model_id: str
    category: str
    prompt: str
    response: str
    auto_scores: Dict
    latency: float
    user_rating: Optional[int] = None  # 1-5, filled by human evaluator
    user_feedback: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelEvalSummary:
    """Summary of all evaluations for one model."""
    model_id: str
    mental_state_accuracy: float     # 0.0 - 1.0
    mental_state_close_accuracy: float  # includes partial credit
    conversation_avg_score: float     # 0.0 - 1.0
    avg_latency: float               # seconds
    avg_user_rating: float            # 1-5
    total_tests: int
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================
# BUILD PROMPTS FOR EVALUATION
# ============================================

def build_mental_state_prompt(scenario: Dict, user_context: str = "") -> str:
    """Build the prompt to send to the LLM for mental state detection."""
    states_str = ", ".join(MENTAL_STATES)
    if user_context:
        return MENTAL_STATE_WITH_CONTEXT_PROMPT.format(
            user_context=user_context,
            user_message=scenario["user_message"],
            states=states_str,
        )
    return MENTAL_STATE_DETECTION_PROMPT.format(
        user_message=scenario["user_message"],
        states=states_str,
    )


def build_conversation_system_prompt(user_context: str = "") -> str:
    """Build a system prompt for conversation evaluation."""
    base = "You are a friendly and helpful AI assistant."
    if user_context:
        base += f" {user_context}"
    return base


# ============================================
# AGGREGATE RESULTS
# ============================================

def aggregate_mental_state_results(results: List[MentalStateResult]) -> Dict:
    """Compute aggregate statistics from mental state results."""
    if not results:
        return {"accuracy": 0, "count": 0}

    correct = sum(1 for r in results if r.is_correct)
    close = sum(1 for r in results if r.score >= 0.5)
    total = len(results)
    avg_latency = sum(r.latency for r in results) / total

    # Per-difficulty breakdown
    by_difficulty = {}
    for r in results:
        # Find difficulty from scenario
        scenario = next(
            (s for s in MENTAL_STATE_SCENARIOS if s["id"] == r.scenario_id), None
        )
        diff = scenario["difficulty"] if scenario else "unknown"
        if diff not in by_difficulty:
            by_difficulty[diff] = {"correct": 0, "total": 0}
        by_difficulty[diff]["total"] += 1
        if r.is_correct:
            by_difficulty[diff]["correct"] += 1

    for diff in by_difficulty:
        d = by_difficulty[diff]
        d["accuracy"] = round(d["correct"] / d["total"], 3) if d["total"] else 0

    return {
        "accuracy": round(correct / total, 3),
        "close_accuracy": round(close / total, 3),
        "correct": correct,
        "total": total,
        "avg_latency": round(avg_latency, 3),
        "by_difficulty": by_difficulty,
    }


def aggregate_conversation_results(results: List[ConversationResult]) -> Dict:
    """Compute aggregate statistics from conversation results."""
    if not results:
        return {"avg_score": 0, "count": 0}

    total = len(results)
    avg_auto = sum(r.auto_scores.get("overall", 0) for r in results) / total
    avg_latency = sum(r.latency for r in results) / total

    rated = [r for r in results if r.user_rating is not None]
    avg_user = sum(r.user_rating for r in rated) / len(rated) if rated else 0

    # Per-category breakdown
    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {"scores": [], "latencies": []}
        by_category[r.category]["scores"].append(r.auto_scores.get("overall", 0))
        by_category[r.category]["latencies"].append(r.latency)

    for cat in by_category:
        c = by_category[cat]
        c["avg_score"] = round(sum(c["scores"]) / len(c["scores"]), 3)
        c["avg_latency"] = round(sum(c["latencies"]) / len(c["latencies"]), 3)
        del c["scores"], c["latencies"]

    return {
        "avg_auto_score": round(avg_auto, 3),
        "avg_user_rating": round(avg_user, 2),
        "avg_latency": round(avg_latency, 3),
        "total": total,
        "user_rated": len(rated),
        "by_category": by_category,
    }
