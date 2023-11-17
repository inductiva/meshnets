"""Tests the data statistics"""
import numpy as np
import pytest

from . import data_statistics

dataset_1 = np.array([[1, 2], [3, 4]])

dataset_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

dataset_3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0], [9, 10, 11,
                                                                 12]])


@pytest.mark.parametrize("dataset", [
    dataset_1,
    dataset_2,
    dataset_3,
])
def test_mean_and_std(dataset):
    """Tests the mean function"""
    for axis in [None, 0, 1]:
        statistics_calculator = data_statistics.OnlineStatistics(axis=axis)
        for e in dataset:
            statistics_calculator.update(e)
        mean = statistics_calculator.get_mean()
        std = statistics_calculator.get_std()
        assert np.allclose(mean, np.mean(dataset, axis=axis, keepdims=True))
        assert np.allclose(std, np.std(dataset, axis=axis, keepdims=True))
