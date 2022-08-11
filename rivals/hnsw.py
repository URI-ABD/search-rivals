import pickle

import hnswlib
import numpy

from .utils import helpers

logger = helpers.make_logger(__name__)


def run():
    dim = 128
    num_elements = 10000

    # Generating sample data
    data = numpy.float32(numpy.random.random((num_elements, dim)))
    ids = numpy.arange(num_elements)

    # Declaring index
    p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Element insertion (can be called several times):
    p.add_items(data, ids)

    # Controlling the recall by setting ef:
    p.set_ef(50)  # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=1)

    # Index objects support pickling
    # WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
    # Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
    p_copy = pickle.loads(pickle.dumps(p))  # creates a copy of index p using pickle round-trip

    # Index parameters are exposed as class properties:
    logger.info(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}")
    logger.info(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
    logger.info(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
    logger.info(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")

    return