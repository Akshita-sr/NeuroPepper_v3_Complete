# backend/services/generative_ai_service.py
# ============================================
# Generative AI Service
# ============================================
# Connects to Stability AI, Runway ML, and local ComfyUI
# for image generation, video generation, and more.
# ============================================

import os
import base64
import httpx
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
STABILITY_BASE_URL = "https://api.stability.ai"

RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY", "")
RUNWAY_BASE_URL = "https://api.dev.runwayml.com/v1"

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# Where to save generated images/videos
OUTPUT_DIR = Path(os.getenv("GENERATED_MEDIA_DIR", "data/generated_media"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class GenerationResult:
    """Result from any generative AI call."""
    success: bool
    provider: str  # "stability", "runway", "comfyui"
    output_type: str  # "image", "video"
    file_path: Optional[str] = None
    base64_data: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_seconds: float = 0.0


# ============================================
# STABILITY AI
# ============================================

class StabilityAIService:
    """
    Connects to Stability AI's REST API.
    Supports: text-to-image, image-to-image, upscale, inpainting.
    
    Get your API key at: https://platform.stability.ai/account/keys
    Pricing: Pay-per-generation (credits system)
    """

    def __init__(self):
        self.api_key = STABILITY_API_KEY
        self.base_url = STABILITY_BASE_URL

    def is_available(self) -> bool:
        """Check if Stability AI is configured."""
        return bool(self.api_key)

    async def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        model: str = "sd3.5-large",  # Options: sd3.5-large, sd3.5-medium, sd3-turbo
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg_scale: float = 7.0,
        seed: int = 0,
    ) -> GenerationResult:
        """
        Generate an image from a text description.
        
        Args:
            prompt: What you want to see (e.g., "a robot in a garden")
            negative_prompt: What you DON'T want (e.g., "blurry, dark")
            model: Which Stable Diffusion model to use
            width/height: Image size in pixels (must be multiples of 64)
            steps: More steps = better quality but slower (20-50)
            cfg_scale: How closely to follow the prompt (1-20, 7 is default)
            seed: Random seed (0 = random)
        """
        if not self.is_available():
            return GenerationResult(
                success=False, provider="stability", output_type="image",
                error="Stability AI API key not set. Add STABILITY_API_KEY to your .env file."
            )

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                # SD3.5 uses the new v2beta endpoint
                response = await client.post(
                    f"{self.base_url}/v2beta/stable-image/generate/sd3",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json",
                    },
                    data={
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "model": model,
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "cfg_scale": cfg_scale,
                        "seed": seed,
                        "output_format": "png",
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    image_b64 = result.get("image", result.get("images", [{}])[0].get("image", ""))
                    
                    # Save to file
                    filename = f"stability_{int(time.time())}.png"
                    filepath = OUTPUT_DIR / filename
                    with open(filepath, "wb") as f:
                        f.write(base64.b64decode(image_b64))

                    return GenerationResult(
                        success=True, provider="stability", output_type="image",
                        file_path=str(filepath), base64_data=image_b64,
                        latency_seconds=time.time() - start,
                        metadata={"model": model, "prompt": prompt, "seed": seed}
                    )
                else:
                    return GenerationResult(
                        success=False, provider="stability", output_type="image",
                        error=f"Stability AI error {response.status_code}: {response.text}",
                        latency_seconds=time.time() - start,
                    )
        except Exception as e:
            return GenerationResult(
                success=False, provider="stability", output_type="image",
                error=str(e), latency_seconds=time.time() - start,
            )

    async def image_to_image(
        self,
        prompt: str,
        image_path: str,
        strength: float = 0.7,
        model: str = "sd3.5-large",
    ) -> GenerationResult:
        """
        Transform an existing image based on a text prompt.
        
        Args:
            prompt: What to change about the image
            image_path: Path to the source image
            strength: How much to change (0.0 = keep original, 1.0 = completely new)
        """
        if not self.is_available():
            return GenerationResult(
                success=False, provider="stability", output_type="image",
                error="Stability AI API key not set."
            )

        start = time.time()
        try:
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()

            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/v2beta/stable-image/generate/sd3",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json",
                    },
                    data={
                        "prompt": prompt,
                        "model": model,
                        "strength": strength,
                        "output_format": "png",
                        "mode": "image-to-image",
                    },
                    files={"image": ("source.png", image_data, "image/png")},
                )

                if response.status_code == 200:
                    result = response.json()
                    image_b64 = result.get("image", "")
                    
                    filename = f"stability_i2i_{int(time.time())}.png"
                    filepath = OUTPUT_DIR / filename
                    with open(filepath, "wb") as f:
                        f.write(base64.b64decode(image_b64))

                    return GenerationResult(
                        success=True, provider="stability", output_type="image",
                        file_path=str(filepath), base64_data=image_b64,
                        latency_seconds=time.time() - start,
                        metadata={"model": model, "prompt": prompt, "strength": strength}
                    )
                else:
                    return GenerationResult(
                        success=False, provider="stability", output_type="image",
                        error=f"Error {response.status_code}: {response.text}",
                        latency_seconds=time.time() - start,
                    )
        except Exception as e:
            return GenerationResult(
                success=False, provider="stability", output_type="image",
                error=str(e), latency_seconds=time.time() - start,
            )

    async def list_available_models(self) -> List[str]:
        """List Stability AI models."""
        return [
            "sd3.5-large",        # Best quality, slowest
            "sd3.5-medium",       # Good balance
            "sd3-turbo",          # Fastest
            "stable-image-core",  # Stability's hosted model
        ]


# ============================================
# RUNWAY ML
# ============================================

class RunwayMLService:
    """
    Connects to Runway ML's Gen-3 API for video generation.
    
    Get your API key at: https://dev.runwayml.com
    Pricing: Per-second of video generated
    """

    def __init__(self):
        self.api_key = RUNWAY_API_KEY
        self.base_url = RUNWAY_BASE_URL

    def is_available(self) -> bool:
        """Check if Runway ML is configured."""
        return bool(self.api_key)

    async def text_to_video(
        self,
        prompt: str,
        duration: int = 5,  # seconds (5 or 10)
        model: str = "gen3a_turbo",
    ) -> GenerationResult:
        """
        Generate a video from a text description.
        
        Args:
            prompt: What you want to see in the video
            duration: Video length in seconds (5 or 10)
            model: "gen3a_turbo" (fast) or "gen4_turbo" (newest)
        """
        if not self.is_available():
            return GenerationResult(
                success=False, provider="runway", output_type="video",
                error="Runway ML API key not set. Add RUNWAY_API_KEY to your .env file."
            )

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                # Step 1: Submit generation task
                response = await client.post(
                    f"{self.base_url}/image_to_video",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Runway-Version": "2024-11-06",
                    },
                    json={
                        "model": model,
                        "promptText": prompt,
                        "duration": duration,
                        "watermark": False,
                    },
                )

                if response.status_code not in (200, 201):
                    return GenerationResult(
                        success=False, provider="runway", output_type="video",
                        error=f"Runway API error {response.status_code}: {response.text}",
                        latency_seconds=time.time() - start,
                    )

                task_id = response.json().get("id")

                # Step 2: Poll until the video is ready
                for _ in range(120):  # Max 10 minutes of polling
                    await self._sleep(5)
                    status_resp = await client.get(
                        f"{self.base_url}/tasks/{task_id}",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "X-Runway-Version": "2024-11-06",
                        },
                    )
                    status_data = status_resp.json()
                    task_status = status_data.get("status", "")

                    if task_status == "SUCCEEDED":
                        video_url = status_data.get("output", [None])[0]
                        if video_url:
                            # Download the video
                            video_resp = await client.get(video_url)
                            filename = f"runway_{int(time.time())}.mp4"
                            filepath = OUTPUT_DIR / filename
                            with open(filepath, "wb") as f:
                                f.write(video_resp.content)

                            return GenerationResult(
                                success=True, provider="runway", output_type="video",
                                file_path=str(filepath),
                                latency_seconds=time.time() - start,
                                metadata={"model": model, "prompt": prompt, "duration": duration, "task_id": task_id}
                            )

                    elif task_status == "FAILED":
                        return GenerationResult(
                            success=False, provider="runway", output_type="video",
                            error=f"Runway generation failed: {status_data.get('failure', 'Unknown error')}",
                            latency_seconds=time.time() - start,
                        )

                return GenerationResult(
                    success=False, provider="runway", output_type="video",
                    error="Runway generation timed out after 10 minutes",
                    latency_seconds=time.time() - start,
                )

        except Exception as e:
            return GenerationResult(
                success=False, provider="runway", output_type="video",
                error=str(e), latency_seconds=time.time() - start,
            )

    async def image_to_video(
        self,
        prompt: str,
        image_path: str,
        duration: int = 5,
        model: str = "gen3a_turbo",
    ) -> GenerationResult:
        """
        Animate a still image into a video.
        
        Args:
            prompt: Description of the motion/animation
            image_path: Path to the source image
            duration: Video length in seconds
        """
        if not self.is_available():
            return GenerationResult(
                success=False, provider="runway", output_type="video",
                error="Runway ML API key not set."
            )

        start = time.time()
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

            # Determine media type
            ext = Path(image_path).suffix.lower()
            media_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
            data_uri = f"data:{media_type};base64,{image_b64}"

            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{self.base_url}/image_to_video",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Runway-Version": "2024-11-06",
                    },
                    json={
                        "model": model,
                        "promptImage": data_uri,
                        "promptText": prompt,
                        "duration": duration,
                        "watermark": False,
                    },
                )

                if response.status_code not in (200, 201):
                    return GenerationResult(
                        success=False, provider="runway", output_type="video",
                        error=f"Error {response.status_code}: {response.text}",
                        latency_seconds=time.time() - start,
                    )

                task_id = response.json().get("id")

                # Poll for completion (same as text_to_video)
                for _ in range(120):
                    await self._sleep(5)
                    status_resp = await client.get(
                        f"{self.base_url}/tasks/{task_id}",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "X-Runway-Version": "2024-11-06",
                        },
                    )
                    status_data = status_resp.json()

                    if status_data.get("status") == "SUCCEEDED":
                        video_url = status_data.get("output", [None])[0]
                        if video_url:
                            video_resp = await client.get(video_url)
                            filename = f"runway_i2v_{int(time.time())}.mp4"
                            filepath = OUTPUT_DIR / filename
                            with open(filepath, "wb") as f:
                                f.write(video_resp.content)

                            return GenerationResult(
                                success=True, provider="runway", output_type="video",
                                file_path=str(filepath),
                                latency_seconds=time.time() - start,
                                metadata={"model": model, "prompt": prompt, "task_id": task_id}
                            )
                    elif status_data.get("status") == "FAILED":
                        return GenerationResult(
                            success=False, provider="runway", output_type="video",
                            error=f"Generation failed: {status_data.get('failure')}",
                            latency_seconds=time.time() - start,
                        )

                return GenerationResult(
                    success=False, provider="runway", output_type="video",
                    error="Timed out", latency_seconds=time.time() - start,
                )

        except Exception as e:
            return GenerationResult(
                success=False, provider="runway", output_type="video",
                error=str(e), latency_seconds=time.time() - start,
            )

    @staticmethod
    async def _sleep(seconds):
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)


# ============================================
# COMFYUI (LOCAL - uses your GPU)
# ============================================

class ComfyUIService:
    """
    Connects to your LOCAL ComfyUI installation for free image generation.
    No API key needed — uses YOUR GPU!
    
    You already have ComfyUI installed at:
    C:\\Users\\akshi\\OneDrive\\Desktop\\ComfyUI_windows_portable\\
    
    Start it with: run_nvidia_gpu.bat (or run_cpu.bat)
    Then it runs at: http://127.0.0.1:8188
    """

    def __init__(self):
        self.base_url = COMFYUI_URL

    async def is_available(self) -> bool:
        """Check if ComfyUI is running locally."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/system_stats")
                return response.status_code == 200
        except Exception:
            return False

    async def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        checkpoint: str = "",
    ) -> GenerationResult:
        """
        Generate an image using your local ComfyUI + GPU.
        
        This is FREE (no API costs) but uses your computer's GPU.
        
        Args:
            prompt: What you want to see
            negative_prompt: What to avoid
            checkpoint: Model name (leave empty for default)
            seed: -1 for random
        """
        start = time.time()

        # Build the ComfyUI workflow JSON
        # This is a standard txt2img workflow
        workflow = {
            "3": {  # KSampler
                "inputs": {
                    "seed": seed if seed >= 0 else int(time.time()),
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSampler",
            },
            "4": {  # Load Checkpoint
                "inputs": {
                    "ckpt_name": checkpoint or "v1-5-pruned-emaonly.safetensors",
                },
                "class_type": "CheckpointLoaderSimple",
            },
            "5": {  # Empty Latent Image
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "6": {  # Positive Prompt
                "inputs": {"text": prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
            },
            "7": {  # Negative Prompt
                "inputs": {"text": negative_prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode",
            },
            "8": {  # VAE Decode
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
                "class_type": "VAEDecode",
            },
            "9": {  # Save Image
                "inputs": {"filename_prefix": "NeuroPepper", "images": ["8", 0]},
                "class_type": "SaveImage",
            },
        }

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                # Queue the prompt
                response = await client.post(
                    f"{self.base_url}/prompt",
                    json={"prompt": workflow},
                )

                if response.status_code != 200:
                    return GenerationResult(
                        success=False, provider="comfyui", output_type="image",
                        error=f"ComfyUI error: {response.text}",
                        latency_seconds=time.time() - start,
                    )

                prompt_id = response.json().get("prompt_id")

                # Poll for completion
                for _ in range(120):
                    await RunwayMLService._sleep(2)
                    history_resp = await client.get(
                        f"{self.base_url}/history/{prompt_id}"
                    )
                    history = history_resp.json()

                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        # Find the SaveImage output
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                image_info = node_output["images"][0]
                                img_filename = image_info["filename"]
                                subfolder = image_info.get("subfolder", "")

                                # Download the image from ComfyUI
                                params = {"filename": img_filename}
                                if subfolder:
                                    params["subfolder"] = subfolder
                                img_resp = await client.get(
                                    f"{self.base_url}/view", params=params
                                )

                                # Save locally
                                filename = f"comfyui_{int(time.time())}.png"
                                filepath = OUTPUT_DIR / filename
                                with open(filepath, "wb") as f:
                                    f.write(img_resp.content)

                                image_b64 = base64.b64encode(img_resp.content).decode()

                                return GenerationResult(
                                    success=True, provider="comfyui", output_type="image",
                                    file_path=str(filepath), base64_data=image_b64,
                                    latency_seconds=time.time() - start,
                                    metadata={"prompt": prompt, "checkpoint": checkpoint}
                                )

                return GenerationResult(
                    success=False, provider="comfyui", output_type="image",
                    error="ComfyUI generation timed out",
                    latency_seconds=time.time() - start,
                )

        except httpx.ConnectError:
            return GenerationResult(
                success=False, provider="comfyui", output_type="image",
                error=(
                    "Cannot connect to ComfyUI. "
                    "Start it by running run_nvidia_gpu.bat in your "
                    "ComfyUI_windows_portable folder."
                ),
                latency_seconds=time.time() - start,
            )
        except Exception as e:
            return GenerationResult(
                success=False, provider="comfyui", output_type="image",
                error=str(e), latency_seconds=time.time() - start,
            )

    async def list_checkpoints(self) -> List[str]:
        """List available model checkpoints in ComfyUI."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/object_info/CheckpointLoaderSimple")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("CheckpointLoaderSimple", {}).get(
                        "input", {}
                    ).get("required", {}).get("ckpt_name", [[]])[0]
        except Exception:
            pass
        return []


# ============================================
# UNIFIED SERVICE (one function for everything)
# ============================================

# Create singleton instances
stability_service = StabilityAIService()
runway_service = RunwayMLService()
comfyui_service = ComfyUIService()


async def check_providers() -> Dict[str, bool]:
    """Check which providers are available."""
    return {
        "stability_ai": stability_service.is_available(),
        "runway_ml": runway_service.is_available(),
        "comfyui_local": await comfyui_service.is_available(),
    }


async def generate_image(
    prompt: str,
    provider: str = "auto",
    **kwargs,
) -> GenerationResult:
    """
    Generate an image using the best available provider.
    
    provider options: "stability", "comfyui", "auto"
    "auto" tries ComfyUI first (free), then Stability AI.
    """
    if provider == "auto":
        # Try free local option first
        if await comfyui_service.is_available():
            return await comfyui_service.text_to_image(prompt, **kwargs)
        elif stability_service.is_available():
            return await stability_service.text_to_image(prompt, **kwargs)
        else:
            return GenerationResult(
                success=False, provider="none", output_type="image",
                error="No image generator available. Start ComfyUI or add STABILITY_API_KEY to .env"
            )
    elif provider == "stability":
        return await stability_service.text_to_image(prompt, **kwargs)
    elif provider == "comfyui":
        return await comfyui_service.text_to_image(prompt, **kwargs)
    else:
        return GenerationResult(
            success=False, provider=provider, output_type="image",
            error=f"Unknown provider: {provider}. Use 'stability', 'comfyui', or 'auto'"
        )


async def generate_video(
    prompt: str,
    provider: str = "runway",
    **kwargs,
) -> GenerationResult:
    """
    Generate a video. Currently only Runway ML supports this.
    """
    if provider == "runway":
        return await runway_service.text_to_video(prompt, **kwargs)
    else:
        return GenerationResult(
            success=False, provider=provider, output_type="video",
            error=f"Unknown video provider: {provider}. Use 'runway'"
        )
