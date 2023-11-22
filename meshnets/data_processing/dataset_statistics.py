"""File to compute dataset statistics"""
import collections.abc

import numpy as np
import npstreams


def _yield_array(array, axis):
    if axis is None:
        for number in array.flatten():
            yield number
    else:
        yield array


def make_iterator(dataset, key, axis, max_iterations):
    """Make iterator over dataset"""
    if isinstance(dataset, collections.abc.Iterable):
        for i, example in enumerate(dataset):
            if i >= max_iterations:
                break
            array = np.array(example[key])
            yield from _yield_array(array, axis)
    elif isinstance(dataset, collections.abc.Sized):
        for i in range(len(dataset)):
            if i >= max_iterations:
                break
            array = np.array(dataset[i][key])
            yield from _yield_array(array, axis)
    else:
        raise ValueError('Dataset must be iterable or sized')


def compute_mean_and_std(dataset, key, axis, max_iterations=100):
    """Compute mean and standard deviation of a dataset

    Args:
        dataset: Dataset to compute statistics on
        key: Key to extract from dataset
        axis: Axis to compute mean and std over
        max_iterations: Maximum number of iterations to use

    Returns:
        mean: Mean of dataset
        std: Standard deviation of dataset
    """
    iterator = make_iterator(dataset, key, axis, max_iterations)
    mean, var = npstreams.average_and_var(iterator, axis=axis)
    return mean, np.sqrt(var)
