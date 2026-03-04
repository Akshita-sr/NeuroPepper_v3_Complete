#!/usr/bin/env python3
# cli/llm_benchmark.py
# ============================================
# LLM Benchmark CLI Tool
# ============================================
# A command-line tool (no GUI) that:
# - Connects to any LLM via REST API
# - Sends multiple prompts one by one
# - Prints the responses
# - Measures response time
# - Saves results to JSON
#
# Usage:
#   python cli/llm_benchmark.py --models ollama/qwen2.5:7b openai/gpt-4o --task conversation
#   python cli/llm_benchmark.py --models all --task mental_state
#   python cli/llm_benchmark.py --list-models
#   python cli/llm_benchmark.py --check-availability
# ============================================

import sys
import os
import json
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.multi_llm_service import (
    call_model, call_models_parallel, get_available_models,
    discover_ollama_models, check_model_availability, LLMResponse,
)
from backend.core.evaluation_engine import (
    MENTAL_STATE_SCENARIOS, CONVERSATION_PROMPTS, MENTAL_STATES,
    build_mental_state_prompt, build_conversation_system_prompt,
    parse_mental_state_response, score_mental_state,
    auto_score_conversation, MentalStateResult, ConversationResult,
    aggregate_mental_state_results, aggregate_conversation_results,
)
from backend.core.user_profile_db import get_user, list_users


# ============================================
# COLORS FOR TERMINAL OUTPUT
# ============================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def cprint(text, color=Colors.END):
    print(f"{color}{text}{Colors.END}")


# ============================================
# LIST MODELS
# ============================================

async def cmd_list_models():
    """List all configured models."""
    cprint("\n=== Available Models ===\n", Colors.HEADER)

    # Static models from config
    models = get_available_models()
    by_provider = {}
    for m in models:
        if m.provider not in by_provider:
            by_provider[m.provider] = []
        by_provider[m.provider].append(m)

    for provider, mlist in by_provider.items():
        cprint(f"\n  {provider.upper()}:", Colors.BOLD)
        for m in mlist:
            print(f"    {m.model_id:<40} {m.display_name}")

    # Discover actually installed Ollama models
    cprint("\n  OLLAMA (actually installed):", Colors.CYAN)
    installed = await discover_ollama_models()
    if installed:
        for m in installed:
            print(f"    {m}")
    else:
        print("    (Could not connect to Ollama or no models installed)")

    print()


# ============================================
# CHECK AVAILABILITY
# ============================================

async def cmd_check_availability(model_ids: list):
    """Check which models are actually reachable."""
    cprint("\n=== Checking Model Availability ===\n", Colors.HEADER)

    if not model_ids or model_ids == ["all"]:
        # Check all configured models
        model_ids = [m.model_id for m in get_available_models()]
        # Also add installed Ollama models
        installed = await discover_ollama_models()
        model_ids = list(set(model_ids + installed))

    for mid in model_ids:
        print(f"  Checking {mid:<40} ... ", end="", flush=True)
        result = await check_model_availability(mid)
        if result["available"]:
            cprint(f"✅ OK ({result['latency']})", Colors.GREEN)
        else:
            cprint(f"❌ FAILED: {result.get('error', 'unknown')}", Colors.RED)

    print()


# ============================================
# CUSTOM PROMPTS MODE
# ============================================

async def cmd_custom_prompts(model_ids: list, prompts: list, system_prompt: str = ""):
    """Send custom prompts to models and print responses."""
    cprint("\n=== Custom Prompt Benchmark ===\n", Colors.HEADER)

    for i, prompt in enumerate(prompts, 1):
        cprint(f"\n--- Prompt {i}/{len(prompts)} ---", Colors.BOLD)
        cprint(f"  \"{prompt}\"\n", Colors.CYAN)

        for mid in model_ids:
            print(f"  [{mid}] ", end="", flush=True)
            result = await call_model(mid, prompt, system_prompt=system_prompt)

            if result.success:
                cprint(f"({result.latency_seconds:.2f}s)", Colors.GREEN)
                # Print response (truncated if very long)
                text = result.response_text.strip()
                if len(text) > 500:
                    text = text[:500] + "... [truncated]"
                print(f"  Response: {text}\n")
            else:
                cprint(f"FAILED: {result.error}", Colors.RED)
                print()


# ============================================
# MENTAL STATE EVALUATION
# ============================================

async def cmd_mental_state_eval(model_ids: list, user_id: str = None,
                                 scenarios: list = None):
    """Run mental-state detection evaluation on models."""
    cprint("\n=== Mental-State Detection Evaluation ===\n", Colors.HEADER)

    # Get user context if provided
    user_context = ""
    if user_id:
        user = get_user(user_id)
        if user:
            user_context = user.to_system_context()
            cprint(f"  Using user profile: {user.name} ({user_id})", Colors.CYAN)

    test_scenarios = scenarios or MENTAL_STATE_SCENARIOS
    cprint(f"  Testing {len(test_scenarios)} scenarios on {len(model_ids)} models\n",
           Colors.CYAN)

    all_results = {}  # model_id -> list of MentalStateResult

    for mid in model_ids:
        cprint(f"\n  Model: {mid}", Colors.BOLD)
        results = []

        for scenario in test_scenarios:
            prompt = build_mental_state_prompt(scenario, user_context)

            # Call the model
            llm_resp = await call_model(mid, prompt, max_tokens=200)

            if llm_resp.success:
                # Parse what the model detected
                parsed = parse_mental_state_response(llm_resp.response_text)
                detected = parsed["detected_state"]
                correct = scenario["correct_state"]
                scoring = score_mental_state(detected, correct)

                result = MentalStateResult(
                    scenario_id=scenario["id"],
                    model_id=mid,
                    user_message=scenario["user_message"],
                    correct_state=correct,
                    detected_state=detected,
                    confidence=parsed["confidence"],
                    is_correct=scoring["correct"],
                    score=scoring["score"],
                    latency=llm_resp.latency_seconds,
                    raw_response=llm_resp.response_text[:300],
                )
                results.append(result)

                # Print result
                icon = "✅" if scoring["correct"] else ("🟡" if scoring["close"] else "❌")
                print(f"    {icon} [{scenario['id']}] "
                      f"Expected: {correct:<10} Got: {detected:<10} "
                      f"({llm_resp.latency_seconds:.2f}s)")
            else:
                print(f"    ❌ [{scenario['id']}] FAILED: {llm_resp.error}")

        all_results[mid] = results

        # Print model summary
        if results:
            agg = aggregate_mental_state_results(results)
            cprint(f"\n    Accuracy: {agg['accuracy']*100:.1f}%  "
                   f"(with partial credit: {agg['close_accuracy']*100:.1f}%)  "
                   f"Avg latency: {agg['avg_latency']:.2f}s", Colors.GREEN)

    # Print comparison table
    _print_mental_state_comparison(all_results)
    return all_results


# ============================================
# CONVERSATION EVALUATION
# ============================================

async def cmd_conversation_eval(model_ids: list, user_id: str = None,
                                 prompts: list = None):
    """Run conversation quality evaluation on models."""
    cprint("\n=== Conversation Quality Evaluation ===\n", Colors.HEADER)

    # Get user context if provided
    system_prompt = build_conversation_system_prompt()
    if user_id:
        user = get_user(user_id)
        if user:
            system_prompt = build_conversation_system_prompt(user.to_system_context())
            cprint(f"  Using user profile: {user.name} ({user_id})", Colors.CYAN)

    test_prompts = prompts or CONVERSATION_PROMPTS
    cprint(f"  Testing {len(test_prompts)} prompts on {len(model_ids)} models\n",
           Colors.CYAN)

    all_results = {}

    for mid in model_ids:
        cprint(f"\n  Model: {mid}", Colors.BOLD)
        results = []

        for conv in test_prompts:
            prompt_text = conv["prompt"]
            llm_resp = await call_model(mid, prompt_text,
                                         system_prompt=system_prompt)

            if llm_resp.success:
                auto_scores = auto_score_conversation(
                    llm_resp.response_text,
                    conv.get("quality_criteria", [])
                )

                result = ConversationResult(
                    prompt_id=conv["id"],
                    model_id=mid,
                    category=conv["category"],
                    prompt=prompt_text,
                    response=llm_resp.response_text,
                    auto_scores=auto_scores,
                    latency=llm_resp.latency_seconds,
                )
                results.append(result)

                score = auto_scores.get("overall", 0)
                icon = "✅" if score > 0.7 else ("🟡" if score > 0.4 else "❌")
                print(f"    {icon} [{conv['id']}] {conv['category']:<15} "
                      f"Score: {score:.2f}  ({llm_resp.latency_seconds:.2f}s)")

                # Print first 150 chars of response
                preview = llm_resp.response_text.strip()[:150]
                print(f"       → {preview}...")
            else:
                print(f"    ❌ [{conv['id']}] FAILED: {llm_resp.error}")

        all_results[mid] = results

        if results:
            agg = aggregate_conversation_results(results)
            cprint(f"\n    Avg auto score: {agg['avg_auto_score']:.2f}  "
                   f"Avg latency: {agg['avg_latency']:.2f}s", Colors.GREEN)

    _print_conversation_comparison(all_results)
    return all_results


# ============================================
# INTERACTIVE USER EVALUATION
# ============================================

async def cmd_interactive_eval(model_ids: list):
    """Let a human rate model responses interactively."""
    cprint("\n=== Interactive User Evaluation ===\n", Colors.HEADER)
    cprint("  Type your prompts, rate responses 1-5, type 'quit' to stop.\n",
           Colors.CYAN)

    results = []
    while True:
        prompt = input(f"\n{Colors.BOLD}Your prompt (or 'quit'): {Colors.END}")
        if prompt.lower() in ("quit", "exit", "q"):
            break

        for mid in model_ids:
            print(f"\n  [{mid}] generating...", end="", flush=True)
            resp = await call_model(mid, prompt)

            if resp.success:
                cprint(f" ({resp.latency_seconds:.2f}s)", Colors.GREEN)
                print(f"\n  Response:\n  {resp.response_text.strip()}\n")

                # Get rating
                try:
                    rating = int(input(f"  Rate 1-5 (1=terrible, 5=excellent): "))
                    rating = max(1, min(5, rating))
                except ValueError:
                    rating = 3
                    print("  (defaulting to 3)")

                feedback = input("  Feedback (optional, press Enter to skip): ").strip()

                results.append({
                    "model": mid,
                    "prompt": prompt,
                    "response": resp.response_text[:500],
                    "latency": resp.latency_seconds,
                    "user_rating": rating,
                    "feedback": feedback,
                })
            else:
                cprint(f" FAILED: {resp.error}", Colors.RED)

    if results:
        _save_results(results, "interactive_eval")
    return results


# ============================================
# FULL EVALUATION (both tasks)
# ============================================

async def cmd_full_eval(model_ids: list, user_id: str = None):
    """Run both mental-state and conversation evaluation."""
    cprint("\n" + "="*60, Colors.HEADER)
    cprint("  FULL MODEL EVALUATION", Colors.HEADER)
    cprint("="*60, Colors.HEADER)

    ms_results = await cmd_mental_state_eval(model_ids, user_id)
    conv_results = await cmd_conversation_eval(model_ids, user_id)

    # Combined summary
    cprint("\n" + "="*60, Colors.HEADER)
    cprint("  COMBINED RESULTS SUMMARY", Colors.HEADER)
    cprint("="*60, Colors.HEADER)

    print(f"\n  {'Model':<35} {'MS Accuracy':>12} {'Conv Score':>12} {'Avg Latency':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")

    for mid in model_ids:
        ms = aggregate_mental_state_results(ms_results.get(mid, []))
        cv = aggregate_conversation_results(conv_results.get(mid, []))
        ms_acc = f"{ms.get('accuracy', 0)*100:.1f}%"
        cv_score = f"{cv.get('avg_auto_score', 0):.2f}"
        all_latencies = [r.latency for r in ms_results.get(mid, [])] + \
                       [r.latency for r in conv_results.get(mid, [])]
        avg_lat = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        print(f"  {mid:<35} {ms_acc:>12} {cv_score:>12} {avg_lat:>11.2f}s")

    # Save full results
    combined = {
        "timestamp": datetime.now().isoformat(),
        "models": model_ids,
        "mental_state": {
            mid: [r.to_dict() for r in results]
            for mid, results in ms_results.items()
        },
        "conversation": {
            mid: [r.to_dict() for r in results]
            for mid, results in conv_results.items()
        },
    }
    _save_results(combined, "full_eval")


# ============================================
# HELPERS
# ============================================

def _print_mental_state_comparison(all_results):
    """Print a comparison table of mental state results."""
    cprint("\n\n=== Mental-State Detection Comparison ===\n", Colors.HEADER)
    print(f"  {'Model':<35} {'Accuracy':>10} {'Partial':>10} {'Avg Time':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    for mid, results in all_results.items():
        agg = aggregate_mental_state_results(results)
        print(f"  {mid:<35} {agg['accuracy']*100:>9.1f}% "
              f"{agg['close_accuracy']*100:>9.1f}% "
              f"{agg['avg_latency']:>9.2f}s")


def _print_conversation_comparison(all_results):
    """Print a comparison table of conversation results."""
    cprint("\n\n=== Conversation Quality Comparison ===\n", Colors.HEADER)
    print(f"  {'Model':<35} {'Auto Score':>12} {'Avg Time':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*10}")

    for mid, results in all_results.items():
        agg = aggregate_conversation_results(results)
        print(f"  {mid:<35} {agg['avg_auto_score']:>11.2f} "
              f"{agg['avg_latency']:>9.2f}s")


def _save_results(data, name_prefix: str):
    """Save results to a JSON file."""
    output_dir = PROJECT_ROOT / "data" / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{name_prefix}_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    cprint(f"\n  📁 Results saved to: {filepath}", Colors.GREEN)


def _resolve_model_ids(raw_ids: list) -> list:
    """
    Resolve 'all' keyword and discover Ollama models.
    """
    if not raw_ids:
        return []

    if "all" in raw_ids:
        models = get_available_models()
        return [m.model_id for m in models]

    return raw_ids


# ============================================
# MAIN CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM Benchmark CLI - Compare language models on conversation and mental-state detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python cli/llm_benchmark.py --list-models

  # Check which models are reachable
  python cli/llm_benchmark.py --check

  # Run mental-state evaluation on two models
  python cli/llm_benchmark.py --models ollama/qwen2.5:7b openai/gpt-4o --task mental_state

  # Run conversation evaluation
  python cli/llm_benchmark.py --models ollama/qwen2.5:7b --task conversation

  # Run full evaluation (both tasks) on all models
  python cli/llm_benchmark.py --models all --task full

  # Send custom prompts
  python cli/llm_benchmark.py --models ollama/qwen2.5:7b --prompts "What is AI?" "Tell me a joke"

  # Interactive mode: rate responses yourself
  python cli/llm_benchmark.py --models ollama/qwen2.5:7b openai/gpt-4o --task interactive

  # Use a user profile for context
  python cli/llm_benchmark.py --models ollama/qwen2.5:7b --task full --user user_adult_01
        """
    )

    parser.add_argument("--models", nargs="+", default=[],
                        help="Model IDs to test (e.g., ollama/qwen2.5:7b openai/gpt-4o). Use 'all' for all.")
    parser.add_argument("--task", choices=["mental_state", "conversation", "full", "interactive", "custom"],
                        default="full", help="Which evaluation to run")
    parser.add_argument("--prompts", nargs="+", default=[],
                        help="Custom prompts to send (for --task custom)")
    parser.add_argument("--system-prompt", default="",
                        help="System prompt to use for custom prompts")
    parser.add_argument("--user", default=None,
                        help="User profile ID for context (e.g., user_adult_01)")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models")
    parser.add_argument("--check", action="store_true",
                        help="Check model availability")
    parser.add_argument("--seed-users", action="store_true",
                        help="Create sample user profiles in the database")

    args = parser.parse_args()

    # Handle special commands
    if args.list_models:
        asyncio.run(cmd_list_models())
        return

    if args.check:
        asyncio.run(cmd_check_availability(args.models))
        return

    if args.seed_users:
        from backend.core.user_profile_db import seed_sample_users
        seed_sample_users()
        return

    # Resolve model IDs
    model_ids = _resolve_model_ids(args.models)
    if not model_ids:
        cprint("❌ No models specified. Use --models or --list-models to see options.", Colors.RED)
        parser.print_help()
        return

    # Run the selected task
    if args.task == "mental_state":
        asyncio.run(cmd_mental_state_eval(model_ids, args.user))
    elif args.task == "conversation":
        asyncio.run(cmd_conversation_eval(model_ids, args.user))
    elif args.task == "full":
        asyncio.run(cmd_full_eval(model_ids, args.user))
    elif args.task == "interactive":
        asyncio.run(cmd_interactive_eval(model_ids))
    elif args.task == "custom":
        if not args.prompts:
            cprint("❌ No prompts given. Use --prompts 'prompt 1' 'prompt 2'", Colors.RED)
            return
        asyncio.run(cmd_custom_prompts(model_ids, args.prompts, args.system_prompt))


if __name__ == "__main__":
    main()
