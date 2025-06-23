import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """_summary_
    Read YAML file and convert it into a ConfigBox object.

    Args:
        path_to_yaml (Path): _description_

    
    Raises:
        BoxValueError: If the YAML file cannot be parsed.
        ValueError: If the YAML file does not contain a dictionary.
        e: empty file error
        
    Returns:
        ConfigBox : ConfigBox type
    """
    
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml. file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"YAML file is empty with path : {path_to_yaml}")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    create list of directories 
    
    Args:
        path_to_directory (list): list of directories.
        ignore_log(bool, optional) : ignore if multiple directories are created. Defaults to False.
    """ 
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at path : {path}")
            
@ensure_annotations
def get_size(path: Path) ->str:
    """get size of file in kbs

    Args:
        path (Path): Path of the files

    Returns:
        str: =size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"