# ── Base image ────────────────────────────────────────────────────────────────
# Use slim Python 3.11 image to keep the final image as small as possible.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy only the files needed for the package install first so Docker can cache
# this expensive layer and only rebuild it when requirements change.
COPY requirements.txt setup.py README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy the rest of the application ─────────────────────────────────────────
COPY config/    ./config/
COPY params.yaml .
COPY app.py      .
COPY main.py     .

# ── Copy trained model artifacts ──────────────────────────────────────────────
# Uncomment after training is complete and artifacts are committed with Git LFS.
COPY artifacts/model_trainer/pegasus-samsum-model/ \
        ./artifacts/model_trainer/pegasus-samsum-model/
COPY artifacts/model_trainer/tokenizer/ \
        ./artifacts/model_trainer/tokenizer/

# ── Non-root user (security best practice) ───────────────────────────────────
RUN useradd -m -u 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
# • Hugging Face Spaces: must be 7860
# • Render / Railway / Fly.io: inject PORT env-var and the app picks it up
# • Oracle Cloud / local: default falls back to 7860 as well
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
# Docker / orchestrators will mark the container unhealthy if /health fails.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-7860}/health')"

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
