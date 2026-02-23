from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import subprocess
import sys
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline

# ── Shared state ──────────────────────────────────────────────────────────────
_pipeline: PredictionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and keep it in memory."""
    global _pipeline
    _pipeline = PredictionPipeline()
    _pipeline.load_model()          # pre-warms tokenizer + model into RAM
    yield
    _pipeline = None                # clean up on shutdown


app = FastAPI(
    title="AI Text Summarizer",
    version="1.0.0",
    description="Abstractive text summarization powered by Pegasus/T5.",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["root"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["health"])
async def health_check():
    """Lightweight endpoint for UptimeRobot / keep-alive pings."""
    return {"status": "healthy", "model_loaded": _pipeline is not None}


@app.get("/train", tags=["training"])
async def training():
    """Kick off the full training pipeline (long-running)."""
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            return Response(f"Training failed!\n{result.stderr}", status_code=500)
        return Response("Training completed successfully!")
    except subprocess.TimeoutExpired:
        return Response("Training timed out!", status_code=504)
    except Exception as e:
        return Response(f"Error: {e}", status_code=500)


@app.post("/predict", response_model=SummarizeResponse, tags=["inference"])
async def predict_route(request: SummarizeRequest):
    """Summarize the provided text."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    try:
        summary = _pipeline.predict(request.text)
        return SummarizeResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # PORT env-var lets each platform (HF Spaces=7860, Render=10000, etc.) inject its own port
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)