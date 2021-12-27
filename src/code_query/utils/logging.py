"""
Helper module for logging
"""
import logging
import logging.config

from code_query.utils.serialize import yml_load


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger
    """ 
    log_config = yml_load("./config/logging.yml")
    logging.config.dictConfig(log_config)
    return logging.getLogger(name)
