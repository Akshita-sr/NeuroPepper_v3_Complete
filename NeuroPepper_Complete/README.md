# NeuroPepper v3.0 — Intelligent Robot AI Platform + LLM Evaluation Suite

> A complete platform for (1) making SoftBank Pepper robots brilliant with modern AI and (2) comparing multiple Language Models on conversation quality and mental-state detection.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## What's In This Project?

This project combines two major parts:

### Part A: NeuroPepper (Pepper Robot AI Platform)
The split-brain architecture that makes Pepper robot incredibly smart by connecting it to modern AI systems (LLMs, vision, RAG, memory). Pepper uses Python 2.7 (NAOqi SDK), so all AI runs on a separate server (Python 3.11+).

**Key Features:**
- Advanced RAG for document Q&A about Pepper
- Multimodal AI (vision, speech, text)
- Persistent memory (robot remembers users)
- Emotional intelligence
- LangGraph-powered agentic behaviors
- Safety-validated robot commands
- Real-time web dashboard

### Part B: LLM Evaluation Suite (NEW in v3.0)
A complete toolkit for your thesis comparing multiple Language Models:

**Key Features:**
- CLI tool to benchmark any LLM (no GUI needed)
- Support for DeepSeek (local + cloud), GPT-4/4o, Claude, and ALL Ollama models
- Mental-state detection evaluation (15 test scenarios)
- Conversation quality evaluation (10 test categories)
- Response time measurement
- User profile database for contextual evaluation
- Interactive human evaluation mode
- REST API for evaluation from any client
- Results saved to JSON for analysis

---

## Project Structure

```
NeuroPepper/
├── backend/                         # All server code
│   ├── app.py                      # Main FastAPI application (v3.0)
│   ├── core/
│   │   ├── rag_manager.py          # RAG document processing
│   │   ├── memory_manager.py       # Persistent robot memory
│   │   ├── performance_monitor.py  # System monitoring
│   │   ├── evaluation_engine.py    # [NEW] Mental-state & conversation scoring
│   │   └── user_profile_db.py      # [NEW] User profiles database
│   ├── routers/
│   │   ├── text_router.py          # Text generation API
│   │   ├── vision_router.py        # Image analysis API
│   │   ├── rag_router.py           # Document Q&A API
│   │   ├── pepper_router.py        # Robot control API
│   │   ├── memory_router.py        # Memory management API
│   │   ├── benchmark_router.py     # Simple model benchmarks
│   │   ├── evaluation_router.py    # [NEW] Full LLM evaluation API
│   │   ├── models_router.py        # Available models listing
│   │   └── ...                     # Other routers
│   ├── services/
│   │   ├── ollama_service.py       # Local Ollama LLM integration
│   │   ├── multi_llm_service.py    # [NEW] Unified multi-LLM service
│   │   ├── speech_service.py       # Speech processing
│   │   ├── vision_service.py       # Computer vision
│   │   └── sound_service.py        # Audio processing
│   ├── agents/
│   │   └── orchestrator.py         # LangGraph orchestrator
│   └── pepper/
│       └── action_library.py       # Safe robot action primitives
├── cli/                             # [NEW] Command-line tools
│   ├── __init__.py
│   └── llm_benchmark.py            # [NEW] Main CLI benchmark tool
├── frontend/                        # Web interface
│   ├── index.html                  # Main dashboard + chat
│   ├── pepper_control.html         # Robot control panel
│   ├── rag_studio.html             # Document upload/query
│   ├── memory_explorer.html        # Memory system explorer
│   └── admin.html                  # Admin dashboard
├── pepper_client/                   # Code that runs ON Pepper (Python 2.7)
│   └── bridge_server.py            # Flask server on Pepper
├── docs/                            # Documentation
│   └── DEEPSEEK_SERVER_SETUP.md    # [NEW] How to set up local DeepSeek
├── data/                            # Runtime data
│   ├── uploads/                    # Uploaded documents
│   ├── vectorstores/               # RAG vector indices
│   ├── memory/                     # Robot memory database
│   ├── user_profiles/              # [NEW] User profile database
│   └── eval_results/               # [NEW] Evaluation result JSONs
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── run_server.py                    # Easy server launcher
└── README.md                        # This file
```

---

## Quick Start

### Step 1: Install Python 3.11+

```bash
# Linux
sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip

# Mac
brew install python@3.11

# Windows: download from https://www.python.org/downloads/
```

### Step 2: Set Up the Project

```bash
cd NeuroPepper
python -m venv venv

# Activate the virtual environment
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (see Configuration section below)
```

### Step 4: Install and Start Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh    # Linux
brew install ollama                               # Mac

# Start Ollama server
ollama serve

# Pull models (in a new terminal)
ollama pull qwen2.5:7b          # Main language model
ollama pull deepseek-r1:7b      # DeepSeek for evaluation
ollama pull llava:latest         # Vision model
ollama pull nomic-embed-text     # Embedding model for RAG
```

### Step 5: Start NeuroPepper Server

```bash
python run_server.py
# Server starts at http://localhost:5000
# API docs at http://localhost:5000/api/docs
```

### Step 6: Run LLM Benchmark (CLI)

```bash
# List available models
python cli/llm_benchmark.py --list-models

# Run full evaluation on one model
python cli/llm_benchmark.py --models ollama/qwen2.5:7b --task full

# Compare multiple models
python cli/llm_benchmark.py --models ollama/qwen2.5:7b ollama/deepseek-r1:7b --task full
```

---

## Configuration (.env file)

```bash
# === LOCAL AI SERVER (REQUIRED) ===
OLLAMA_BASE_URL=http://localhost:11434

# === CLOUD AI SERVICES (fill the ones you have) ===
OPENAI_API_KEY=sk-your-key-here        # For GPT-4 / GPT-4o
ANTHROPIC_API_KEY=your-key-here         # For Claude
DEEPSEEK_API_KEY=your-key-here          # For DeepSeek Cloud

# === DEEPSEEK LOCAL SERVER ===
# (Only if running DeepSeek on a separate lab server)
# See docs/DEEPSEEK_SERVER_SETUP.md for setup
DEEPSEEK_LOCAL_URL=http://LAB_SERVER_IP:8000/v1

# === PEPPER ROBOT ===
PEPPER_IP=192.168.1.100
PEPPER_PORT=9559

# === SERVER ===
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
```

---

## CLI Benchmark Tool — Complete Usage

The CLI tool runs from the terminal with no GUI. It sends prompts to models via REST API and prints the responses.

### Basic Commands

```bash
# See all options
python cli/llm_benchmark.py --help

# List all models
python cli/llm_benchmark.py --list-models

# Check which models are actually reachable
python cli/llm_benchmark.py --check

# Create sample user profiles for testing
python cli/llm_benchmark.py --seed-users
```

### Running Evaluations

```bash
# Mental-state detection only
python cli/llm_benchmark.py --models ollama/qwen2.5:7b --task mental_state

# Conversation quality only
python cli/llm_benchmark.py --models ollama/qwen2.5:7b --task conversation

# Full evaluation (both tasks)
python cli/llm_benchmark.py --models ollama/qwen2.5:7b --task full

# Compare multiple models at once
python cli/llm_benchmark.py \
    --models ollama/qwen2.5:7b ollama/deepseek-r1:7b openai/gpt-4o \
    --task full

# Evaluate ALL configured models
python cli/llm_benchmark.py --models all --task full
```

### Custom Prompts

```bash
# Send your own prompts to models
python cli/llm_benchmark.py \
    --models ollama/qwen2.5:7b openai/gpt-4o \
    --task custom \
    --prompts "What is AI?" "Tell me a joke" "Explain gravity simply"
```

### User Profile Context

```bash
# First create sample users
python cli/llm_benchmark.py --seed-users

# Run with user context (the model knows who it's talking to)
python cli/llm_benchmark.py \
    --models ollama/qwen2.5:7b \
    --task full \
    --user user_child_01
```

### Interactive Human Evaluation

```bash
# Rate responses yourself in real-time
python cli/llm_benchmark.py \
    --models ollama/qwen2.5:7b openai/gpt-4o \
    --task interactive
```

### Where Results Are Saved

All results are automatically saved as JSON in `data/eval_results/`.

---

## Supported Models

| Model ID | Provider | Needs API Key? | Notes |
|---|---|---|---|
| `ollama/qwen2.5:7b` | Ollama (local) | No | Best general-purpose local model |
| `ollama/qwen2.5:3b` | Ollama (local) | No | Faster, less accurate |
| `ollama/deepseek-r1:7b` | Ollama (local) | No | DeepSeek via Ollama |
| `ollama/deepseek-r1:14b` | Ollama (local) | No | Larger DeepSeek |
| `ollama/llama3.1:8b` | Ollama (local) | No | Meta LLaMA 3.1 |
| `ollama/mistral:7b` | Ollama (local) | No | Mistral AI |
| `ollama/gemma2:9b` | Ollama (local) | No | Google Gemma 2 |
| `ollama/phi3:mini` | Ollama (local) | No | Microsoft Phi-3 |
| `deepseek_local/deepseek-chat` | Lab server | No | Your own DeepSeek server |
| `deepseek_cloud/deepseek-chat` | DeepSeek Cloud | Yes | Cloud API |
| `openai/gpt-4o` | OpenAI | Yes | GPT-4o (latest) |
| `openai/gpt-4-turbo` | OpenAI | Yes | GPT-4 Turbo |
| `openai/gpt-4` | OpenAI | Yes | GPT-4 |
| `openai/gpt-3.5-turbo` | OpenAI | Yes | GPT-3.5 (cheaper) |
| `claude/claude-sonnet-4-20250514` | Anthropic | Yes | Claude Sonnet 4 |
| `claude/claude-haiku-4-5-20251001` | Anthropic | Yes | Claude Haiku 4.5 |

You can also use ANY model installed in Ollama by its name.

---

## Mental-State Detection Evaluation

Tests how well each model detects emotional/mental states from text.

**10 states:** happy, sad, angry, anxious, tired, confused, excited, bored, stressed, neutral

**15 scenarios** at easy/medium/hard difficulty. Scoring gives full credit for exact match, half credit for close match (e.g. "stressed" for "anxious"), and zero for wrong.

**Example output:**
```
Model: ollama/qwen2.5:7b
    ✅ [ms_01] Expected: tired      Got: tired      (1.23s)
    ✅ [ms_02] Expected: happy      Got: happy      (0.98s)
    🟡 [ms_03] Expected: angry      Got: stressed   (1.45s)
    ❌ [ms_04] Expected: anxious    Got: sad        (1.12s)

    Accuracy: 73.3%  (with partial credit: 80.0%)  Avg latency: 1.19s
```

---

## Conversation Quality Evaluation

Tests 10 categories: greeting, knowledge, empathy, humor, follow-up, emotional support, factual, creative, reasoning, and child-friendly.

Automatic scoring measures length, coherence, engagement, and error-free output. Human evaluation (interactive mode) adds 1-5 ratings.

---

## User Profile Database

Give models context about who they're talking to. Pre-made profiles include adults, teens, children, and seniors with different preferences.

```bash
python cli/llm_benchmark.py --seed-users   # Create sample profiles
python cli/llm_benchmark.py --models ollama/qwen2.5:7b --task full --user user_child_01
```

---

## REST API Endpoints (New in v3.0)

All existing NeuroPepper endpoints still work. New evaluation endpoints at `/eval/`:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/eval/models` | List all configured models |
| POST | `/eval/models/check` | Check model availability |
| POST | `/eval/compare` | Compare models on one prompt |
| POST | `/eval/eval/mental-state` | Run mental-state evaluation |
| POST | `/eval/eval/conversation` | Run conversation evaluation |
| GET | `/eval/users` | List user profiles |
| POST | `/eval/users` | Create user profile |
| POST | `/eval/eval/rate` | Submit human rating |
| GET | `/eval/eval/summary` | Get evaluation summary |

Full interactive docs at `http://localhost:5000/api/docs`.

---

## Setting Up DeepSeek on Your Lab Server

See `docs/DEEPSEEK_SERVER_SETUP.md` for three options:

1. **Ollama** (easiest, 5 min) — `ollama pull deepseek-r1:7b && ollama serve`
2. **vLLM** (best performance) — Uses HuggingFace models directly
3. **llama.cpp** (lightest resources) — Runs GGUF quantized models

---

## For Your Thesis

### Suggested Workflow

1. Set up models (Ollama + cloud API keys)
2. Seed user profiles: `python cli/llm_benchmark.py --seed-users`
3. Run automated evaluations: `python cli/llm_benchmark.py --models <models> --task full`
4. Run user studies: `python cli/llm_benchmark.py --models <models> --task interactive`
5. Collect results from `data/eval_results/`
6. Analyze with Python/pandas

### Key Contributions This Supports

1. Split-brain architecture bypassing Python 2.7 limitations
2. Unified multi-LLM service (one interface for all providers)
3. Mental-state detection benchmark (15 scenarios, 3 difficulty levels)
4. Conversation quality evaluation framework
5. User profile database for contextual AI testing
6. Advanced RAG pipeline for technical documentation
7. Persistent memory system for personalized interaction
8. Safety-validated action library for physical robot control

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Ollama not running" | Open new terminal, run `ollama serve` |
| "Model not found" | Run `ollama pull <model-name>` first |
| "OPENAI_API_KEY not set" | Add key to `.env` file |
| "Connection refused" for DeepSeek | Check lab server is running, URL is correct |
| "CUDA out of memory" | Use smaller models (3b instead of 7b) |
| CLI shows 0 models | Check Ollama is running and/or API keys are set |

---

**Made with care for Pepper Robot researchers and LLM evaluation**
