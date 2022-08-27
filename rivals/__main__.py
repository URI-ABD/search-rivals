import logging

from rivals import faiss_bench
from rivals import hnsw_bench
from rivals.utils import dataset
from rivals.utils import helpers
from rivals.utils import paths

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = helpers.make_logger('main', 'INFO')

NAMES = {
    "fashion-mnist-784-euclidean",
    "fashion-mnist",
    "sift",
    "gist",
}

for name in NAMES:
    logger.info('')

    data_path = paths.DATA_ROOT.joinpath(f'{name}.hdf5')
    if not data_path.exists():
        logger.info(f'Could not find file at {data_path}. Skipping ...')
        continue

    logger.info(f'Reading {name} data ...')
    train, test, true_idx, true_dist = dataset.get_dataset(data_path)
    logger.info(f'Using {name} data with shape {train.shape} and {test.shape[0]} queries ...')

    hnsw_bench.run(train, test, true_idx, true_dist, 100)
    # faiss_nc at 1 is perfect recall
    # larger values speed up queries at the cost of recall
    # even 2 is at best 93.6% recall on fashion-mnist
    faiss_nc = 1 # perfect recall
    faiss_bench.run(train, test, true_idx, true_dist, 100, faiss_nc)

logger.info('')
logger.info('Done!')

