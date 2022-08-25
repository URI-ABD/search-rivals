import h5py
import numpy

from .paths import DATA_ROOT


def get_dataset():
    fmnist = "fashion-mnist-784-euclidean.hdf5"
    data_path = DATA_ROOT.joinpath(fmnist)
    f = h5py.File(data_path, 'r')
    # keys: ['distances', 'neighbors', 'test', 'train']
    train = numpy.asarray(f['train'], dtype=numpy.float32)
    test = numpy.asarray(f['test'], dtype=numpy.float32)
    true_indices = numpy.asarray(f['neighbors'], dtype=numpy.int32)
    true_distances = numpy.asarray(f['distances'], dtype=numpy.float32)
    return train, test, true_indices, true_distances