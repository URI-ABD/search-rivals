import time
import numpy
import faiss

from .utils import helpers

logger = helpers.make_logger(__name__)


def run(train, test, k):
    n, d = train.shape
    test_n, test_d = test.shape
    assert(test_d == d)
    ids = numpy.arange(n)
    
    # begin indexing
    indexing_start = time.perf_counter()
    index = faiss.IndexFlatL2(d) # build the index
    index.add(train)
    indexing_elapsed = time.perf_counter() - indexing_start
    # done indexing
    
    # begin search
    search_start = time.perf_counter()
    D, I = index.search(test, k) # search returns distances and indices
    search_elapsed = time.perf_counter() - search_start
    # done searching


    logger.info(f"Parameters passed to constructor:  dim={d}")
    # logger.info(f"Search speed/quality trade-off parameter: ef={p.ef}")
    logger.info(f"Indexing time:{indexing_elapsed = :.2e}")
    logger.info(f"Search time:{search_elapsed = :.2e}")
    logger.info(f"Search time per query:{search_elapsed/test_n = :.2e}")