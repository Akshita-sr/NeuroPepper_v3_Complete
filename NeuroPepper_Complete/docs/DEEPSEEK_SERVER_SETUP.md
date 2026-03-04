# DeepSeek Local Server Setup Guide
# ============================================
# How to install and run DeepSeek on your lab
# computer so it works as a REST API server.
# ============================================

## Option 1: Using Ollama (Easiest)

Ollama can run DeepSeek models locally with ONE command.
This is the recommended approach for your lab server.

### Step 1: Install Ollama

```bash
# Linux (your lab server)
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows
# Download from https://ollama.ai/download
```

### Step 2: Pull a DeepSeek Model

```bash
# DeepSeek-R1 7B (recommended - needs ~6GB RAM + 4GB VRAM)
ollama pull deepseek-r1:7b

# DeepSeek-R1 14B (better quality, needs ~12GB RAM + 8GB VRAM)
ollama pull deepseek-r1:14b

# DeepSeek-R1 32B (best quality, needs ~24GB RAM + 16GB VRAM)
ollama pull deepseek-r1:32b
```

### Step 3: Start the Server

```bash
# Start Ollama (it listens on port 11434 by default)
ollama serve
```

### Step 4: Make It Accessible on the Network

By default Ollama only listens on localhost. To let other
computers in the lab connect:

```bash
# Set the host to 0.0.0.0 to listen on all interfaces
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Or as a systemd service (permanent):
sudo systemctl edit ollama
# Add these lines:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

### Step 5: Test the API

```bash
# From the lab server itself
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-r1:7b",
  "prompt": "Hello, what is 2+2?",
  "stream": false
}'

# From another computer on the network
curl http://LAB_SERVER_IP:11434/api/generate -d '{
  "model": "deepseek-r1:7b",
  "prompt": "Hello!",
  "stream": false
}'
```

### Step 6: Configure NeuroPepper

In your `.env` file:
```bash
OLLAMA_BASE_URL=http://LAB_SERVER_IP:11434
```

Now any `ollama/deepseek-r1:7b` model call from NeuroPepper
will use your lab server!


## Option 2: Using vLLM (Higher Performance)

vLLM is faster for serving large models but takes more setup.

### Step 1: Install vLLM

```bash
pip install vllm
```

### Step 2: Download the Model

```bash
# This downloads from HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
"
```

### Step 3: Start the Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

### Step 4: Test

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

### Step 5: Configure NeuroPepper

In your `.env` file:
```bash
DEEPSEEK_LOCAL_URL=http://LAB_SERVER_IP:8000/v1
```


## Option 3: Using llama.cpp Server (Lightweight)

Good if you have limited resources.

### Step 1: Install llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc)  # or: cmake -B build && cmake --build build
```

### Step 2: Download a GGUF Model

```bash
# From HuggingFace
wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
```

### Step 3: Start Server

```bash
./llama-server \
    -m DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    -c 4096 \
    -ngl 99  # offload all layers to GPU
```

### Step 4: Configure NeuroPepper

Same as Option 2:
```bash
DEEPSEEK_LOCAL_URL=http://LAB_SERVER_IP:8000/v1
```


## Quick Comparison

| Method     | Setup Time | Performance | Memory Usage | OpenAI Compatible |
|-----------|-----------|------------|-------------|-------------------|
| Ollama     | 5 min      | Good        | Medium       | Yes (via proxy)    |
| vLLM       | 30 min     | Excellent   | Higher       | Yes (native)       |
| llama.cpp  | 45 min     | Good        | Lowest       | Yes (native)       |

## Recommended: Start with Ollama

For your thesis, Ollama is the easiest path. It gives you:
- DeepSeek running locally in 5 minutes
- Automatic GPU acceleration
- The same REST API format that NeuroPepper already supports
- Easy model switching (just `ollama pull` another model)
