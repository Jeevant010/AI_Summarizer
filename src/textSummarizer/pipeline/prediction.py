from pathlib import Path

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from transformers import pipeline as hf_pipeline


class PredictionPipeline:
    """Wraps the summarisation model. Call load_model() once to pre-warm,
    then call predict() for every inference request.  The pipeline object is
    cached on the instance so the model is only loaded from disk once."""

    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self._pipe = None           # lazy-loaded / pre-warmed via load_model()

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
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self._pipe = hf_pipeline(
            "summarization",
            model=model_source,
            tokenizer=tokenizer,
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