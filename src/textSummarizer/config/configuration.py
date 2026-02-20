from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import (DataIngestionConfig,
                                   DataValidationConfig,
                                   DataTransformationConfig,
                                   ModelTrainerConfig,
                                   ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: str = CONFIG_FILE_PATH,
        params_filepath: str = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])
        
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config =self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = Path(config.root_dir),
            source_URL = config.source_URL,
            local_data_file = Path(config.local_data_file),
            unzip_dir = Path(config.unzip_dir),
        )
        
        return data_ingestion_config
    
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
            data_dir=Path(config.data_dir),
        )
        
        return data_validation_config
    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config= self.config.data_transformation
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            tokenizer_name=config.tokenizer_name,  # if this is a path, use Path(); if just a model name, keep as str
            dev_run=getattr(config, 'dev_run', False),
            dev_model=getattr(config, 'dev_model', None),
        )
        
        return data_transformation_config
    
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=str(config.model_ckpt),
            num_train_epochs=int(params.num_train_epochs),
            warmup_steps=int(params.warmup_steps),
            per_device_train_batch_size=int(params.per_device_train_batch_size),
            weight_decay=float(params.weight_decay),
            logging_steps=int(params.logging_steps),
            eval_strategy=str(params.eval_strategy),
            eval_steps=int(params.eval_steps),
            save_steps=int(float(params.save_steps)),
            gradient_accumulation_steps=int(params.gradient_accumulation_steps)
        )

        # Optional dev quick-train settings (not all configs will have these)
        try:
            model_trainer_config = ModelTrainerConfig(
                root_dir=Path(config.root_dir),
                data_path=Path(config.data_path),
                model_ckpt=str(config.model_ckpt),
                num_train_epochs=int(params.num_train_epochs),
                warmup_steps=int(params.warmup_steps),
                per_device_train_batch_size=int(params.per_device_train_batch_size),
                weight_decay=float(params.weight_decay),
                logging_steps=int(params.logging_steps),
                eval_strategy=str(params.eval_strategy),
                eval_steps=int(params.eval_steps),
                save_steps=int(float(params.save_steps)),
                gradient_accumulation_steps=int(params.gradient_accumulation_steps),
                dev_run=getattr(config, 'dev_run', False),
                dev_model=getattr(config, 'dev_model', None),
                dev_subset=int(getattr(config, 'dev_subset', 0))
            )
        except Exception:
            pass
        
        return model_trainer_config
    
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            metric_file_name = config.metric_file_name
           
        )

        return model_evaluation_config