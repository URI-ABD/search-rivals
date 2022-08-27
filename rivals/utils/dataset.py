import pathlib

import h5py
import numpy


def get_dataset(data_path: pathlib.Path):

    with h5py.File(data_path, 'r') as f:
        # keys for fmnist: ['distances', 'neighbors', 'test', 'train']

        train = numpy.asarray(f['train'], dtype=numpy.float32)
        test = numpy.asarray(f['test'], dtype=numpy.float32)
        true_indices = numpy.asarray(f['neighbors'], dtype=numpy.int32)
        true_distances = numpy.asarray(f['distances'], dtype=numpy.float32)

    return train, test, true_indices, true_distances
