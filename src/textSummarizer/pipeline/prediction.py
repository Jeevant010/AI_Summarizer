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
    def load_model(self) -> None:
        """Load the tokenizer and model into memory (call once at startup)."""
        logger.info("Loading summarisation model from %s", self.config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))
        self._pipe = hf_pipeline(
            "summarization",
            model=str(self.config.model_path),
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