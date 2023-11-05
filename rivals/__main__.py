import logging
import sys
import numpy as np

from rivals import faiss_bench
from rivals import hnsw_bench
from rivals import mrpt_bench
# from rivals.utils import dataset
from rivals.utils import helpers
# from rivals.utils import paths

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger('main', 'INFO')

NAMES = {
    "deep-image": "cosine",
    "fashion-mnist": "euclidean",
    "gist": "euclidean",
    "glove-25": "cosine",
    "glove-50": "cosine",
    "glove-100": "cosine",
    "glove-200": "cosine",
    "mnist": "euclidean",
    "sift": "euclidean",
    "lastfm": "cosine",
    "nytimes": "cosine"
}

RIVALS = {
    "faiss",
    "hnsw",
    "mrpt"
}

usage = f"{sys.argv[0]} datadir dataset rival k"
if len(sys.argv) != 5:
    sys.exit(usage)

datadir = sys.argv[1]
dataset = sys.argv[2]
rival = sys.argv[3]
k = int(sys.argv[4])




target = np.load(f"{datadir}/{dataset}-train.npy")
queries = np.load(f"{datadir}/{dataset}-test.npy")

logger.info(f'Using {dataset} data with shape {target.shape} and {queries.shape[0]} queries ...')

if rival == "faiss":
    faiss_bench.run(target, queries, k)
elif rival == "hnsw":
    hnsw_bench.run(target, queries, k)
elif rival == "mrpt":
    mrpt_bench.run(target, queries, k)
else:
    print(f"Unknown rival {rival}")

logger.info('')
logger.info('Done!')

