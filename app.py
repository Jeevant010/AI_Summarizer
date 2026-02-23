from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import subprocess
import threading
import sys
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline

# ── Shared state ──────────────────────────────────────────────────────────────
_pipeline: PredictionPipeline | None = None
_model_ready   = threading.Event()   # set once model is in RAM
_model_error: str | None = None      # non-None if load failed


def _load_model_bg() -> None:
    """Background thread: load model and signal readiness."""
    global _pipeline, _model_error
    try:
        _pipeline = PredictionPipeline()
        _pipeline.load_model()
    except Exception as exc:
        _model_error = str(exc)
    finally:
        _model_ready.set()  # always unblock waiters, even on error


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
            status_code=500,
            content={"status": "error", "detail": _model_error},
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
        raise HTTPException(status_code=500, detail=f"Model failed to load: {_model_error}")
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