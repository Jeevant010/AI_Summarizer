---
title: AI Text Summarizer
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# AI_Summarizer

## Workflows

1. Update Config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in the src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the app.py

## Hugging Face Spaces runtime notes

- The app now sets Hugging Face cache paths at startup:
	- `HF_HOME=/data/cache` (fallback: `/tmp/hf_cache`)
	- `TRANSFORMERS_CACHE=/data/cache`
	- `HF_HUB_CACHE=/data/cache/hub`
- This avoids recurring corrupted tokenizer snapshots in ephemeral `/tmp` cache.

### Useful environment variables

- `MODEL_LOAD_MAX_ATTEMPTS` (default `3`): number of startup retries for model load.
- `MODEL_LOAD_RETRY_DELAY` (default `8`): seconds between retries.
- `RESET_HF_CACHE_ON_START` (default `0`): set to `1` for a one-time cache reset if startup keeps failing.

### Recommended recovery procedure

1. Set `RESET_HF_CACHE_ON_START=1`.
2. Trigger a new Space build/restart.
3. After healthy startup, set `RESET_HF_CACHE_ON_START=0`.
