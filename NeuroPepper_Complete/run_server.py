#!/usr/bin/env python3
"""
NeuroPepper Server Launcher
===========================
Easy-to-use script to start the NeuroPepper server.

Usage:
    python run_server.py
    python run_server.py --port 8000
    python run_server.py --no-reload
"""

import os
import sys
import subprocess
import argparse
import socket
import time
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    """Print the NeuroPepper startup banner."""
    print(f"""
{Colors.CYAN}
    _   __                     ____                            
   / | / /__  __  ___________/ __ \\___  ____  ____  ___  _____
  /  |/ / _ \\/ / / / ___/ __ / /_/ / _ \\/ __ \\/ __ \\/ _ \\/ ___/
 / /|  /  __/ /_/ / /  / /_/ / ____/  __/ /_/ / /_/ /  __/ /    
/_/ |_/\\___/\\__,_/_/   \\____/_/    \\___/ .___/ .___/\\___/_/     
                                       /_/   /_/                
{Colors.END}
{Colors.BOLD}Intelligent Robot AI Platform{Colors.END}
""")


def check_python_version():
    """Ensure Python 3.11+ is being used."""
    if sys.version_info < (3, 11):
        print(f"{Colors.WARNING}⚠️  Python 3.11+ is recommended. You have {sys.version}{Colors.END}")
        print("   Some features may not work correctly.")
        return False
    return True


def check_ollama():
    """Check if Ollama server is running."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get('models', [])
        print(f"{Colors.GREEN}✅ Ollama is running with {len(models)} model(s){Colors.END}")
        return True
    except Exception:
        print(f"{Colors.WARNING}⚠️  Ollama is not running{Colors.END}")
        print("   Start Ollama in another terminal: ollama serve")
        return False


def check_required_models():
    """Check if required models are installed."""
    required = ['qwen2.5', 'nomic-embed-text']
    
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m['name'] for m in response.json().get('models', [])]
        
        missing = []
        for req in required:
            if not any(req in m for m in models):
                missing.append(req)
        
        if missing:
            print(f"{Colors.WARNING}⚠️  Missing recommended models: {', '.join(missing)}{Colors.END}")
            print(f"   Run: ollama pull {missing[0]}")
        else:
            print(f"{Colors.GREEN}✅ Required models are available{Colors.END}")
            
    except Exception:
        pass


def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def create_directories():
    """Create necessary data directories."""
    dirs = [
        'data/uploads',
        'data/vectorstores',
        'data/memory',
        'logs'
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print(f"{Colors.GREEN}✅ Data directories ready{Colors.END}")


def run_server(host='0.0.0.0', port=5000, reload=True, workers=1):
    """Start the FastAPI server."""
    
    if is_port_in_use(port):
        print(f"{Colors.FAIL}❌ Port {port} is already in use{Colors.END}")
        print(f"   Try: python run_server.py --port {port + 1}")
        sys.exit(1)
    
    print(f"\n{Colors.BLUE}🚀 Starting server...{Colors.END}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    
    print(f"\n{Colors.GREEN}{'='*50}")
    print(f"   Open in browser: http://localhost:{port}")
    print(f"   API docs: http://localhost:{port}/api/docs")
    print(f"{'='*50}{Colors.END}\n")
    
    # Build uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.app:socket_app",
        "--host", host,
        "--port", str(port),
    ]
    
    if reload:
        cmd.append("--reload")
    else:
        cmd.extend(["--workers", str(workers)])
    
    # Run uvicorn
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}👋 Server stopped{Colors.END}")


def main():
    parser = argparse.ArgumentParser(description="NeuroPepper Server Launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (production)")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies")
    
    args = parser.parse_args()
    
    print_banner()
    
    print(f"{Colors.BOLD}Checking environment...{Colors.END}\n")
    
    check_python_version()
    create_directories()
    
    ollama_ok = check_ollama()
    if ollama_ok:
        check_required_models()
    
    if args.check_only:
        print(f"\n{Colors.CYAN}Check complete!{Colors.END}")
        return
    
    if not ollama_ok:
        print(f"\n{Colors.WARNING}Starting anyway (some features may not work){Colors.END}")
        time.sleep(2)
    
    run_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
