from typing import Optional, Dict, Any
import logging
import json
import yaml


def read_file(path: str, logger: logging.LogRecord) -> Optional[str]:
    """
    Read a file and return the contents as a string. 
    
    Args:
        path (str): The path to the file to read.
        logger (logging.LogRecord): The logger to use for logging.

    Returns:
        Optional[str]: The contents of the file as a string. None if the file could not be read.
    """
    try:
        with open(path, 'r') as file:
            content: str = file.read()

        return content
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
    
def write_to_file(path: str, content: str, logger: logging.LogRecord) -> None:
    """
    Write content to a file.
    
    Args:
        path (str): The path to the file to write to.
        content (str): The content to write to the file.
        logger (logging.LogRecord): The logger to use for logging.
    """
    try:
        with open(path, 'a', encoding='utf-8') as file:
            file.write(content)
        logger.info(f"Content written to file: {path}")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except Exception as e:
        logger.error(f"Error writing to file '{path}: {e}")