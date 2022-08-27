import time
import numpy
import faiss

from .utils import helpers

logger = helpers.make_logger(__name__)


def run(train, test, true_idx, true_dist, k, nc):
    n, d = train.shape
    test_n, test_d = test.shape
    assert(test_d == d)
    ids = numpy.arange(n)
    
    # begin indexing
    indexing_start = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    # index = faiss.IndexFlatL2(d) # build the flat index
    index = faiss.IndexIVFFlat(quantizer, d, nc, faiss.METRIC_L2)
    index.train(train)
    index.add(train)
    indexing_elapsed = time.perf_counter() - indexing_start
    # done indexing
    
    # begin search
    search_start = time.perf_counter()
    distances, labels = index.search(test, k) # search returns distances and indices
    search_elapsed = time.perf_counter() - search_start
    # done searching

    # Measure recall
    correct = 0
    for i in range(test_n):
        for label in labels[i]:
            for correct_label in true_idx[i]:
                if label == correct_label:
                    correct += 1
                    break


    logger.info(f"Parameters passed to constructor:  dim={d}")
    # logger.info(f"Search speed/quality trade-off parameter: ef={p.ef}")
    logger.info(f"Indexing time:{indexing_elapsed = :.2e}")
    logger.info(f"Search time:{search_elapsed = :.2e}")
    logger.info(f"Search time per query:{search_elapsed/test_n = :.2e}")
    logger.info(f"Recall:{float(correct)/(k*test_n) = :.2e}")