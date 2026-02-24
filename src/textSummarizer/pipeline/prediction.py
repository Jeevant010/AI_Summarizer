from pathlib import Path

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import pipeline as hf_pipeline


class PredictionPipeline:
    """Wraps the summarisation model. Call load_model() once to pre-warm,
    then call predict() for every inference request.  The pipeline object is
    cached on the instance so the model is only loaded from disk once."""

    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self._pipe = None           # lazy-loaded / pre-warmed via load_model()

    # ------------------------------------------------------------------
    @staticmethod
    def _is_hub_source(source) -> bool:
        return isinstance(source, str)

    # ------------------------------------------------------------------
    @staticmethod
    def _is_cache_corruption_error(message: str) -> bool:
        lowered = message.lower()
        return (
            "spiece.model" in lowered
            or "error parsing line" in lowered
            or "sentencepiece" in lowered
        )

    # ------------------------------------------------------------------
    def _build_pipeline(self, model_source, tokenizer_source, force_download: bool = False):
        tokenizer_kwargs = {}
        model_kwargs = {}

        if force_download:
            # force_download is meaningful for Hub IDs; for local paths we keep
            # default loading behaviour.
            if self._is_hub_source(tokenizer_source):
                tokenizer_kwargs["force_download"] = True
            if self._is_hub_source(model_source):
                model_kwargs["force_download"] = True

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_source, **model_kwargs)

        return hf_pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
        )

    # ------------------------------------------------------------------
    def _resolve_model_source(self):
        """Return (model_source, tokenizer_source) as either absolute Path
        objects (local fine-tuned model) or Hub model-ID strings (fallback).

        Using Path objects for local paths is critical on Windows: newer
        huggingface_hub versions validate strings as repo-IDs and reject
        Windows paths that contain backslashes, spaces, or drive letters.
        Path objects are recognised as os.PathLike and bypass that check.
        """
        tokenizer_path = Path(self.config.tokenizer_path).resolve()
        model_path     = Path(self.config.model_path).resolve()

        if tokenizer_path.is_dir() and model_path.is_dir():
            logger.info("Loading local fine-tuned model from %s", model_path)
            return model_path, tokenizer_path

        # Local model not present – fall back to the Hub checkpoint so the
        # app still starts on HuggingFace Spaces or a fresh clone.
        hub_id = self.config.hub_model_id
        logger.warning(
            "Local model not found at %s. "
            "Falling back to HuggingFace Hub model '%s'. "
            "Run the training pipeline (python main.py) to use the fine-tuned model.",
            model_path, hub_id,
        )
        return hub_id, hub_id

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load the tokenizer and model into memory (call once at startup)."""
        model_source, tokenizer_source = self._resolve_model_source()
        try:
            self._pipe = self._build_pipeline(model_source, tokenizer_source)
        except Exception as exc:
            message = str(exc)
            should_retry = (
                self._is_hub_source(model_source)
                and self._is_hub_source(tokenizer_source)
                and self._is_cache_corruption_error(message)
            )

            if not should_retry:
                raise

            logger.warning(
                "Tokenizer/model cache appears corrupted (%s). Retrying with force_download=True.",
                message,
            )
            self._pipe = self._build_pipeline(
                model_source,
                tokenizer_source,
                force_download=True,
            )

        logger.info("Model loaded and ready.")

    # ------------------------------------------------------------------
    def predict(self, text: str) -> str:
        """Return the summary for *text*. Loads the model on first call if not
        already pre-warmed (useful for standalone / testing usage)."""
        if self._pipe is None:
            self.load_model()

        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
        output: str = self._pipe(text, **gen_kwargs)[0]["summary_text"]
        return output