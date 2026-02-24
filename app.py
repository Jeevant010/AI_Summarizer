from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import subprocess
import threading
import time
import sys
import os
import shutil
from pathlib import Path
from starlette.responses import RedirectResponse
from fastapi.responses import Response


def _configure_hf_cache() -> None:
    """Configure Hugging Face cache directory before any model imports.

    On HuggingFace Spaces, `/data` is persistent between restarts and is safer
    than `/tmp` for large model artifacts.
    """
    preferred_cache_root = Path("/data/cache")
    fallback_cache_root = Path("/tmp/hf_cache")
    cache_root = preferred_cache_root if preferred_cache_root.parent.exists() else fallback_cache_root

    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "hub").mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")

    # Optional: set RESET_HF_CACHE_ON_START=1 for a one-time hard cache reset.
    if os.environ.get("RESET_HF_CACHE_ON_START", "0") == "1":
        hub_cache = cache_root / "hub"
        if hub_cache.exists():
            shutil.rmtree(hub_cache, ignore_errors=True)
        hub_cache.mkdir(parents=True, exist_ok=True)


_configure_hf_cache()

from textSummarizer.pipeline.prediction import PredictionPipeline

# ── Shared state ──────────────────────────────────────────────────────────────
_pipeline: PredictionPipeline | None = None
_model_ready   = threading.Event()   # set once model is in RAM
_model_error: str | None = None      # non-None if load failed


def _load_model_bg() -> None:
    """Background thread: load model and signal readiness."""
    global _pipeline, _model_error
    max_attempts = int(os.environ.get("MODEL_LOAD_MAX_ATTEMPTS", "3"))
    retry_delay_seconds = int(os.environ.get("MODEL_LOAD_RETRY_DELAY", "8"))
    recovery_retry_delay_seconds = int(os.environ.get("MODEL_RECOVERY_RETRY_DELAY", "30"))

    _model_error = None
    while True:
        for attempt in range(1, max_attempts + 1):
            try:
                _pipeline = PredictionPipeline()
                _pipeline.load_model()
                _model_error = None
                _model_ready.set()
                return
            except Exception as exc:
                _model_error = str(exc)
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds)

        # Keep retrying in the background instead of getting stuck forever.
        _model_ready.clear()
        time.sleep(recovery_retry_delay_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Kick off model loading in a background thread so the HTTP server
    starts immediately and HuggingFace Spaces passes its liveness check
    while the ~2 GB model is being downloaded / loaded into RAM."""
    t = threading.Thread(target=_load_model_bg, daemon=True, name="model-loader")
    t.start()
    yield
    # Cleanup on shutdown
    global _pipeline
    _pipeline = None
    _model_ready.clear()


app = FastAPI(
    title="AI Text Summarizer",
    version="1.0.0",
    description=(
        "Abstractive text summarization powered by Pegasus.\n\n"
        "**Endpoints**\n"
        "- `POST /predict` — summarize text\n"
        "- `GET  /health`  — liveness / readiness check\n"
        "- `GET  /train`   — re-run the training pipeline"
    ),
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Hannah: Hey, do you have Betty's number?\nAmanda: Nope, sorry. I don't.\nHannah: Oh, I thought you did. Well, can you call her?\nAmanda: I don't have her number I said!"}
            ]
        }
    }


class SummarizeResponse(BaseModel):
    summary: str
    model_status: str = "ready"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["root"], include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["health"])
async def health_check():
    """Liveness + readiness probe.

    Returns HTTP 200 once the model is loaded, HTTP 503 while still loading.
    UptimeRobot should monitor this endpoint every 10-15 minutes to keep the
    HuggingFace Space awake 24/7.
    """
    if _model_error:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "detail": f"Model recovery in progress: {_model_error}",
            },
        )
    if not _model_ready.is_set():
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "detail": "Model is warming up, please retry in a moment."},
        )
    return {"status": "healthy", "model_loaded": True}


@app.get("/train", tags=["training"])
async def training():
    """Kick off the full training pipeline (long-running, runs in subprocess)."""
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            return Response(f"Training failed!\n{result.stderr}", status_code=500)
        return Response("Training completed successfully!")
    except subprocess.TimeoutExpired:
        return Response("Training timed out after 2 hours!", status_code=504)
    except Exception as e:
        return Response(f"Error: {e}", status_code=500)


@app.post("/predict", response_model=SummarizeResponse, tags=["inference"])
async def predict_route(request: SummarizeRequest):
    """Summarize the provided text.

    Returns HTTP 503 while the model is still loading on first startup.
    """
    if _model_error:
        raise HTTPException(
            status_code=503,
            detail=f"Model is recovering from a load failure. Please retry shortly. Last error: {_model_error}",
        )
    if not _model_ready.is_set():
        raise HTTPException(
            status_code=503,
            detail="Model is still warming up. Please retry in a moment.",
        )
    try:
        summary = _pipeline.predict(request.text)
        return SummarizeResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=75,   # prevent HF Spaces proxy from dropping long requests
    )