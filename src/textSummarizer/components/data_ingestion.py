import os
import urllib.request as request
import zipfile
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size
from pathlib import Path
from textSummarizer.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_data(self):
        try:
            if not self.config.local_data_file.exists():
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=str(self.config.local_data_file)
                )
                logger.info(f"{filename} downloaded with info: {headers}")
            else:
                logger.info(f"File already exists of size: {get_size(self.config.local_data_file)}")
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise e
            
    def extract_zip_file(self):
        """
        Extracts the zip file located at `local_data_file` to the directory `unzip_dir`.
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            logger.info(f"Extracting {self.config.local_data_file} to {unzip_path}")
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info("Extraction completed.")
        except Exception as e:
            logger.error(f"Error extracting zip file: {e}")
            raise e
