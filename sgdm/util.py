import numpy as np
from scipy.signal import boxcar, convolve

__version__ = '0.1.0'


def clean_dataset(dataset):
    """Remove any row in dataset for which one or more columns is np.nan
    """

    # Get rid of NaNs
    dataset = dataset[~np.isnan(dataset[:, 1:]).any(axis=1), :]

    return dataset


def boxcar_smooth_dataset(dataset, window_size):
    window = boxcar(window_size)
    return convolve(dataset, window, 'same') / window_size


def datetime2epoch(ts):

    return (ts - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


def nearest(items, pivot, direction=None):
    """Return the element in items that is closest in value to pivot"""
    if direction == 'after':
        return min([i for i in items if i > pivot], key=lambda x: abs(x - pivot))
    elif direction == 'before':
        return min([i for i in items if i > pivot], key=lambda x: abs(x - pivot))

    return min(items, key=lambda x: abs(x - pivot))
