"""Minimal vLLM + Phoenix server (lean response)
------------------------------------------------
* Phoenix tracing – auto local/cloud via `phoenix.otel.register`.
* vLLM engine – LoRA‑enabled, version‑agnostic shim.
* Records latency in the span, but **response includes only the model output**.
"""

from __future__ import annotations

import asyncio
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Dict, Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from huggingface_hub import login, snapshot_download
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from phoenix.otel import register as phoenix_register
from pydantic import BaseModel
from vllm import AsyncLLMEngine, EngineArgs
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

# ────────────────────── Configuration ──────────────────────
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
HF_TOKEN        = os.getenv("HF_TOKEN")
PHOENIX_PROJECT = "vllm-custom-server-final"

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "300"))
BATCH_MAX_TOKENS        = int(os.getenv("BATCH_MAX_TOKENS", "16384"))
GPU_MEMORY_UTIL         = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))

if not PHOENIX_API_KEY:
    raise EnvironmentError("PHOENIX_API_KEY is not set")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is not set")

# ────────────────────── Phoenix tracer ─────────────────────
TRACER: Optional[trace.Tracer] = None

def init_tracer() -> None:
    """Register Phoenix collector (local if reachable, else cloud)."""
    try:
        httpx.get("http://localhost:6006", timeout=1)
        phoenix_register(project_name=PHOENIX_PROJECT, batch=True)
        print("🏠  Using local Phoenix collector")
    except Exception:
        phoenix_register(
            project_name=PHOENIX_PROJECT,
            endpoint="https://app.phoenix.arize.com/v1/traces",
            headers={"api_key": PHOENIX_API_KEY},
            batch=True,
        )
        print("☁️  Using Phoenix Cloud collector")
    global TRACER
    TRACER = trace.get_tracer(__name__)

# ────────────────────── vLLM engine ────────────────────────
LORA_CACHE: Dict[str, LoRARequest] = {}
ENGINE: Optional[AsyncLLMEngine] = None
SEM = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

warnings.filterwarnings("ignore", message="No tokenizer found in.*")
os.environ.setdefault("VLLM_SKIP_LORA_TOKENIZER_CHECK", "1")


def build_engine() -> AsyncLLMEngine:
    """Initialise vLLM + LoRA with compatibility shims."""
    lora_dir = snapshot_download("aliazn/mathchat-mistral")
    LORA_CACHE["mathchat"] = LoRARequest("mathchat", 1, lora_dir)

    args = EngineArgs(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer="mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_mode="mistral",
        dtype="auto",
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_num_batched_tokens=BATCH_MAX_TOKENS,
        max_num_seqs=256,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
    )
    args.disable_log_requests = getattr(args, "disable_log_requests", True)
    args.engine_use_ray       = getattr(args, "engine_use_ray", False)

    return AsyncLLMEngine.from_engine_args(args)

# ────────────────────── FastAPI app ────────────────────────
app = FastAPI(title="vLLM + Phoenix (lean)", lifespan=None)

class InferenceRequest(BaseModel):
    prompt: str
    model: str = "mathchat"
    max_new_tokens: int = 128
    temperature: float = 0
    top_p: float = 0.95
    stream: bool = False

class InferenceResponse(BaseModel):
    response: str

# ────────────────────── Generation core ────────────────────
async def run_generation(req: InferenceRequest) -> str:
    lora = LORA_CACHE.get(req.model)
    sampling = SamplingParams(
        max_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        skip_special_tokens=True,
    )

    span_attrs = {
        SpanAttributes.INPUT_VALUE: req.prompt[:1000],
        "llm.request.model": req.model,
        "llm.max_tokens": req.max_new_tokens,
    }

    with TRACER.start_as_current_span("vllm_generate", attributes=span_attrs) as span:
        start = time.perf_counter()
        text = ""

        async for out in ENGINE.generate(req.prompt, sampling, str(uuid4()), lora):
            if out.finished:
                text = out.outputs[0].text
                break
            if req.stream:
                text = out.outputs[0].text

        span.set_attribute("generation_time_sec", round(time.perf_counter() - start, 3))
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, text[:1000])

    return text

# ────────────────────── Lifespan ───────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    global ENGINE
    print("🚀  Initialising …")
    init_tracer()
    login(HF_TOKEN)
    ENGINE = build_engine()
    print("✅  Ready")
    yield
    print("🛑  Shutdown")

app.router.lifespan_context = lifespan

# ────────────────────── Routes ─────────────────────────────
@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    async with SEM:
        try:
            txt = await run_generation(req)
            return InferenceResponse(response=txt)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

