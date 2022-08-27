import logging

import numpy

from . import constants


def make_logger(name: str, level: str = None):
    logger_ = logging.getLogger(name)
    logger_.setLevel(constants.LOG_LEVEL if level is None else level)
    return logger_


logger = make_logger(__name__)


def _recall_1d(pred: numpy.ndarray, true: numpy.ndarray) -> float:
    assert pred.ndim == 1
    assert true.ndim == 1

    pred = set(pred.flatten())
    true = set(true.flatten())

    tp = len([i for i in true if i in pred])
    fn = len([i for i in true if i not in pred])

    return tp / (tp + fn)


def measure_recall(labels: numpy.ndarray, true_idx: numpy.ndarray) -> float:
    assert labels.ndim == 2
    assert labels.shape == true_idx.shape
    # `labels` and `true_idx` are 2d-arrays with the same shape. Each row
    # contains the k neighbors for its corresponding query.

    recalls = [
        _recall_1d(labels[i], true_idx[i])
        for i in range(labels.shape[0])
    ]

    return sum(recalls) / len(recalls)
