# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment variables ─────────────────────────────────────────────────────
# HF_HOME / TRANSFORMERS_CACHE must point to /tmp so they are writable
# under any user (HuggingFace Spaces restricts writes outside /tmp and /home).
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_DATASETS_CACHE=/tmp/hf_cache/datasets \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY app.py main.py params.yaml ./
COPY config/ ./config/

# ── Create writable runtime directories ──────────────────────────────────────
# Create artifact dirs and the HF cache dir here (as root) so they already
# exist with correct ownership before we switch to the non-root user.
RUN mkdir -p \
        artifacts/model_trainer \
        artifacts/model_evaluation \
        artifacts/data_ingestion \
        artifacts/data_transformation \
        artifacts/data_validation \
        /tmp/hf_cache \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app /tmp/hf_cache

USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
# HuggingFace Spaces requires exactly 7860.
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
# Returns 200 once model is loaded, 503 while still warming up — both are
# acceptable to the orchestrator (non-5xx codes keep the container alive).
HEALTHCHECK --interval=20s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://localhost:7860/health', timeout=8)
    # 200 = ready, 503 = still loading — both mean the server is alive
    sys.exit(0 if r.status in (200, 503) else 1)
except Exception:
    sys.exit(1)
"

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
