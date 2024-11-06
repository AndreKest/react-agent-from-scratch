import json
import logging

import numpy as np

def add(x: int, y:int, logger: logging.LogRecord) -> int:
    """ Add two numbers. """
    logger.info(f"Adding {x} and {y}")
    return x + y

def multiply(x: int, y:int, logger: logging.LogRecord) -> int:
    """ Multiply two numbers. """
    logger.info(f"Multiplying {x} and {y}")
    return x * y

def subtract(x: int, y:int, logger: logging.LogRecord) -> int: 
    """ Subtract from the number x the number y. """
    logger.info(f"Subtracting {y} from {x}")
    return x - y