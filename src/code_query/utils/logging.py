"""
Helper module for logging
"""
import logging
import logging.config
import datetime as dt

from code_query.model.encoder import Encoder
from code_query.utils.serialize import yml_load


log_config = yml_load("./config/logging.yml")
logging.config.dictConfig(log_config)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger
    """ 
    return logging.getLogger(name)


def get_run_name(encoder_type: Encoder.Types, prefix=None) -> str:
    timestamp = dt.datetime.now().strftime("%D-%H:%M")
    return f"{encoder_type.value}-{timestamp}"
