import logging

from . import constants


def make_logger(name: str, level: str = None):
    logger_ = logging.getLogger(name)
    logger_.setLevel(constants.LOG_LEVEL if level is None else level)
    return logger_


logger = make_logger(__name__)
