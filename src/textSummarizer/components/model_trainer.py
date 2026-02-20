from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch 
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Choose model checkpoint: prefer dev_model when dev_run is enabled
        model_checkpoint = str(self.config.model_ckpt)
        if getattr(self.config, 'dev_run', False) and getattr(self.config, 'dev_model', None):
            model_checkpoint = self.config.dev_model
            
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        dataset_samsum_pt = load_from_disk(str(self.config.data_path))
        
        # If dev_subset is set, use only a small subset of data for quick testing
        dev_subset = getattr(self.config, 'dev_subset', 0)
        if dev_subset > 0:
            if "train" in dataset_samsum_pt:
                dataset_samsum_pt["train"] = dataset_samsum_pt["train"].select(range(min(dev_subset, len(dataset_samsum_pt["train"]))))
            if "validation" in dataset_samsum_pt:
                validation_size = min(dev_subset // 4, len(dataset_samsum_pt["validation"]))
                dataset_samsum_pt["validation"] = dataset_samsum_pt["validation"].select(range(validation_size))

        # ensure datasets are returned as torch tensors for efficient dataloading
        if "train" in dataset_samsum_pt:
            try:
                dataset_samsum_pt["train"] = dataset_samsum_pt["train"].with_format("torch")
            except Exception:
                pass
        if "validation" in dataset_samsum_pt:
            try:
                dataset_samsum_pt["validation"] = dataset_samsum_pt["validation"].with_format("torch")
            except Exception:
                pass

        trainer_args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=int(self.config.num_train_epochs),
            warmup_steps=int(self.config.warmup_steps),
            per_device_train_batch_size=int(self.config.per_device_train_batch_size),
            per_device_eval_batch_size=int(self.config.per_device_train_batch_size),
            weight_decay=float(self.config.weight_decay),
            logging_steps=int(self.config.logging_steps),
            eval_strategy=str(self.config.eval_strategy),
            eval_steps=int(self.config.eval_steps),
            save_steps=int(float(getattr(self.config, "save_steps", 1e6))),
            gradient_accumulation_steps=int(self.config.gradient_accumulation_steps),
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=int(getattr(self.config, "dataloader_num_workers", 4)),
            save_total_limit=int(getattr(self.config, "save_total_limit", 3)),
            load_best_model_at_end=bool(getattr(self.config, "load_best_model_at_end", False)),
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )
        
        trainer.train()
        
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))