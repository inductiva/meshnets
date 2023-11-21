"""Tests the data statistics"""
import numpy as np
import pytest

from . import data_statistics

dataset_1 = np.array([[1, 2], [3, 4]])

dataset_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

dataset_3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0], [9, 10, 11,
                                                                 12]])


def _make_expected_targets_for_dataset(dataset):
    return ({
        x: np.mean(dataset, axis=x, keepdims=True) for x in [None, 0, 1]
    }, {
        x: np.std(dataset, axis=x, keepdims=True) for x in [None, 0, 1]
    })


(expected_mean_for_dataset_1,
 expected_std_for_dataset_1) = _make_expected_targets_for_dataset(dataset_1)
(expected_mean_for_dataset_2,
 expected_std_for_dataset_2) = _make_expected_targets_for_dataset(dataset_2)
(expected_mean_for_dataset_3,
 expected_std_for_dataset_3) = _make_expected_targets_for_dataset(dataset_3)

dataset_with_diff_shapes = [[1, 2], [1, 2, 3]]
expected_mean_for_dataset_with_diff_shapes = {
    None: np.array([1.8]),
    1: np.array([[1.5], [2.0]])
}
expected_std_for_dataset_with_diff_shapes = {
    None: np.array([np.std([1, 2, 1, 2, 3])]),
    1: np.array([[np.std([1, 2])], [np.std([1, 2, 3])]])
}


@pytest.mark.parametrize(
    "dataset,expected_mean,expected_std,axis",
    [(dataset_with_diff_shapes, expected_mean_for_dataset_with_diff_shapes,
      expected_std_for_dataset_with_diff_shapes, [None, 1]),
     (dataset_1, expected_mean_for_dataset_1, expected_std_for_dataset_1,
      [None, 0, 1]),
     (dataset_2, expected_mean_for_dataset_2, expected_std_for_dataset_2,
      [None, 0, 1]),
     (dataset_3, expected_mean_for_dataset_3, expected_std_for_dataset_3,
      [None, 0, 1])])
def test_mean_and_std_diff_shapes(dataset, expected_mean, expected_std, axis):
    for a in axis:
        mean, std = _compute_mean_and_std(dataset, a)
        assert np.allclose(mean, expected_mean[a])
        assert np.allclose(std, expected_std[a])


def _compute_mean_and_std(dataset, axis):
    statistics_calculator = data_statistics.OnlineStatistics(axis=axis)
    for e in dataset:
        statistics_calculator.update(e)
    mean = statistics_calculator.get_mean()
    std = statistics_calculator.get_std()
    return mean, std
