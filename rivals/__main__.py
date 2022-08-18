import logging
from .utils import dataset

from rivals import hnsw_bench, faiss_bench

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)

train, test = dataset.get_dataset() # placeholder for now, just fmnist

hnsw_bench.run(train, test, 10)
faiss_bench.run(train, test, 10)


