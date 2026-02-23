import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        # choose tokenizer checkpoint: prefer dev_model when dev_run is enabled
        tokenizer_checkpoint = self.config.tokenizer_name
        if getattr(self.config, 'dev_run', False) and getattr(self.config, 'dev_model', None):
            tokenizer_checkpoint = self.config.dev_model

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        
    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            max_length=512,
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
        save_path = self.config.root_dir / "samsum_dataset"
        
        # Temporarily disabled skip check to force transformation
        # # Check if processed dataset already exists
        # if save_path.exists() and (save_path / "dataset_dict.json").exists():
        #     logger.info(f"Processed dataset already exists at {save_path}. Skipping transformation.")
        #     return
            
        logger.info(f"Loading dataset from {self.config.data_path}")
        dataset_samsum = load_from_disk(str(self.config.data_path))
        logger.info("Tokenizing dataset...")
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        
        # ensure labels are correctly formatted for seq2seq Trainer
        def rename_for_trainer(batch):
            # already returns 'labels' in convert_examples_to_features; keep
            return batch

        dataset_samsum_pt = dataset_samsum_pt.map(rename_for_trainer, batched=True)
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving processed dataset to {save_path}")
        # Use absolute path with proper Windows formatting to handle spaces in path
        abs_save_path = os.path.abspath(str(save_path))
        # Disable multiprocessing on Windows to avoid path issues with spaces
        dataset_samsum_pt.save_to_disk(abs_save_path, num_proc=1)
