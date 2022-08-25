import time
import numpy
import hnswlib

from .utils import helpers

logger = helpers.make_logger(__name__)


def run(train, test, true_idx, true_dist, k):
    
    n, d = train.shape
    test_n, test_d = test.shape

    # Generating sample data
    # data = numpy.float32(numpy.random.random((num_elements, dim)))
    ids = numpy.arange(n)

    # Declaring index
    p = hnswlib.Index(space='l2', dim=d)  # possible options are l2, cosine or ip

    indexing_start = time.perf_counter()
    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=n, ef_construction=1000, M=16)
    indexing_elapsed = time.perf_counter() - indexing_start

    # # Element insertion (can be called several times):
    p.add_items(train, ids)

    # # Controlling the recall by setting ef:
    p.set_ef(n)  # ef should always be > k


    search_start = time.perf_counter()
    # # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(test, k)
    search_elapsed = time.perf_counter() - search_start


    # Measure recall
    # This is all wrong, it doesn't separate labels per query
    correct = 0
    for i in range(test_n): # for each query
        for label in labels[i]:
            for correct_label in true_idx[i]:
                if label == correct_label:
                    correct += 1
                    break

    # todo recall from scikit-learn
    # todo faiss-ivf, faiss-hnsw, faiss...

    # # Index parameters are exposed as class properties:
    logger.info(f"Parameters passed to constructor:  space={p.space}, dim={p.dim}")
    logger.info(f"Index construction: M={p.M}, ef_construction={p.ef_construction}")
    logger.info(f"Index size is {p.element_count} and index capacity is {p.max_elements}")
    logger.info(f"Search speed/quality trade-off parameter: ef={p.ef}")
    logger.info(f"Indexing time:{indexing_elapsed = :.2e}")
    logger.info(f"Search time:{search_elapsed = :.2e}")
    logger.info(f"Search time per query:{search_elapsed/test_n = :.2e}")
    logger.info(f"Recall:{float(correct)/(k*test_n) = :.2e}")

    return
