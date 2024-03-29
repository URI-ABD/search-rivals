import time

import hnswlib
import numpy

from .utils import helpers

logger = helpers.make_logger(__name__)


def run(target, queries, k):
    
    n, d = target.shape
    test_n, test_d = queries.shape

    # Generating sample data
    # data = numpy.float32(numpy.random.random((num_elements, dim)))
    ids = numpy.arange(n)

    # Declaring index
    logger.info(f'Building index ...')
    indexing_start = time.perf_counter()

    p = hnswlib.Index(space='l2', dim=d)  # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=n)

    # # Element insertion (can be called several times):
    p.add_items(target, ids)

    # # Controlling the recall by setting ef:
    p.set_ef(2 * k)  # ef should always be > k

    indexing_elapsed = time.perf_counter() - indexing_start

    logger.info(f"Parameters passed to constructor:  space={p.space}, dim={p.dim}")
    logger.info(f"Index construction: M={p.M}, ef_construction={p.ef_construction}")
    logger.info(f"Index size is {p.element_count} and index capacity is {p.max_elements}")
    logger.info(f"Search speed/quality trade-off parameter: ef={p.ef}")
    logger.info(f"Indexing time: {indexing_elapsed:.2e} s")

    logger.info(f'Starting search ...')
    search_start = time.perf_counter()
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(queries, k)
    search_elapsed = time.perf_counter() - search_start

    logger.info(f"Search time: {search_elapsed:.2e} s")
    logger.info(f"Search time per query: {search_elapsed/test_n:.2e} s")
    logger.info(f"Throughput: { test_n / search_elapsed:.3f}")

    # Measure recall
    # recall = helpers.measure_recall(labels, true_idx)

    # todo recall from scikit-learn
    # todo faiss-ivf, faiss-hnsw, faiss...

    # Index parameters are exposed as class properties:
    # logger.info(f"Recall: {recall:.2e}")

    return
