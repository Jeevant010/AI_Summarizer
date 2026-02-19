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
        # allow dev-run override for a small quick model
        model_checkpoint = self.config.model_ckpt
        if getattr(self.config, 'dev_run', False) and getattr(self.config, 'dev_model', None):
            model_checkpoint = self.config.dev_model

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        dataset_samsum_pt = load_from_disk(str(self.config.data_path))

        # If dev_run is enabled, take small subsets to speed up training
        if getattr(self.config, 'dev_run', False) and getattr(self.config, 'dev_subset', 0) > 0:
            subset = int(self.config.dev_subset)
            for split in ["test", "validation"]:
                if split in dataset_samsum_pt:
                    n = min(subset, len(dataset_samsum_pt[split]))
                    dataset_samsum_pt[split] = dataset_samsum_pt[split].select(range(n))
        
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=1e6,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        
        
        trainer = Trainer(model=model_pegasus, args=trainer_args,
              processing_class=tokenizer, data_collator=seq2seq_data_collator,
              train_dataset=dataset_samsum_pt["test"],
              eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()
        
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))