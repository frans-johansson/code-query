"""
Helper module for logging
"""
import logging
import logging.config

from code_query.utils.serialize import yml_load


log_config = yml_load("./config/logging.yml")
logging.config.dictConfig(log_config)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger
    """ 
    return logging.getLogger(name)
