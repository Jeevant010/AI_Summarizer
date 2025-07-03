import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True
        )
        target_encodings = self.tokenizer(
            text_target=example_batch['summary'],
            max_length=128,
            truncation=True
        )
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
        }
        
    def convert(self):
        logger.info(f"Loading dataset from {self.config.data_path}")
        dataset_samsum = load_from_disk(str(self.config.data_path))
        logger.info("Tokenizing dataset...")
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        save_path = self.config.root_dir / "samsum_dataset"
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving processed dataset to {save_path}")
        dataset_samsum_pt.save_to_disk(str(save_path))
