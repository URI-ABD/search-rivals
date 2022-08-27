import logging
from .utils import dataset

from rivals import hnsw_bench, faiss_bench

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)

train, test, true_idx, true_dist = dataset.get_dataset() # placeholder for now, just fmnist

hnsw_bench.run(train, test, true_idx, true_dist, 100)
# faiss_nc at 1 is perfect recall
# larger values speed up queries at the cost of recall
# even 2 is at best 93.6% recall on fashion-mnist
faiss_nc = 1 # perfect recall
faiss_bench.run(train, test, true_idx, true_dist, 100, faiss_nc)


