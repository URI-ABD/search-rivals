import time

import faiss
import numpy

from .utils import helpers

logger = helpers.make_logger(__name__)


def run(train, test, true_idx, true_dist, k):
    n, d = train.shape
    test_n, test_d = test.shape
    assert (test_d == d)
    ids = numpy.arange(n)

    # begin indexing
    indexing_start = time.perf_counter()
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(train)
    indexing_elapsed = time.perf_counter() - indexing_start
    # done indexing

    logger.info(f"Parameters passed to constructor:  dim={d}")
    logger.info(f"Indexing time: {indexing_elapsed:.2e}")

    # begin search
    search_start = time.perf_counter()
    distances, labels = index.search(test, k)  # search returns distances and indices
    search_elapsed = time.perf_counter() - search_start
    # done searching

    logger.info(f"Total Search time: {search_elapsed:.2e}")
    logger.info(f"Search time per query: {search_elapsed / test_n:.2e}")

    # Measure recall
    recall = helpers.measure_recall(labels, true_idx)
    logger.info(f"Recall: {recall:.2e}")

    return
