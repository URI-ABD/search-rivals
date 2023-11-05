import time

import faiss
import numpy

from .utils import helpers

logger = helpers.make_logger(__name__)

def run(target, queries, k):
# def run(train, test, true_idx, true_dist, k, nc, nprobe):
    n, d = target.shape
    test_n, test_d = queries.shape
    assert (test_d == d)
    ids = numpy.arange(n)
# faiss_nc at 1 is perfect recall
    # larger values speed up queries at the cost of recall
    # even 2 is at best 93.6% recall on fashion-mnist
    nc = 1 # perfect recall
    nprobe = 1 # does not seem to help when nc is 1. Helps recall when nc is higher.
    # begin indexing
    indexing_start = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    # index = faiss.IndexFlatL2(d) # build the flat index
    index = faiss.IndexIVFFlat(quantizer, d, nc, faiss.METRIC_L2)
    index.train(target)
    index.add(target)
    index.nprobe = nprobe
    indexing_elapsed = time.perf_counter() - indexing_start
    # done indexing

    logger.info(f"Parameters passed to constructor:  dim={d}, nc={nc}, nprobe={nprobe}")
    logger.info(f"Indexing time: {indexing_elapsed:.2e}")

    # begin search
    search_start = time.perf_counter()
    distances, labels = index.search(queries, k)  # search returns distances and indices
    # _, distances, labels = index.range_search(queries, 10.0)  # search returns distances and indices
    search_elapsed = time.perf_counter() - search_start
    # done searching

    logger.info(f"Total Search time: {search_elapsed:.2e}")
    logger.info(f"Search time per query: {search_elapsed / test_n:.2e}")
    logger.info(f"Throughput: { test_n / search_elapsed:.3f}")

    # Measure recall
    # recall = helpers.measure_recall(labels, true_idx)
    # logger.info(f"Recall: {recall:.2e}")

    return
