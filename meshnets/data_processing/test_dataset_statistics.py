import pytest

import numpy as np

from . import dataset_statistics


class DatasetWithLen:

    def __init__(self, data):
        self.data = [{'a': example} for example in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetWithIter:

    def __init__(self, data):
        self.data = [{'a': example} for example in data]

    def __iter__(self):
        for example in self.data:
            yield example


def _make_expected_targets_for_dataset(dataset):
    return ({
        x: np.mean(dataset.T, axis=x) for x in [None, 0, 1]
    }, {
        x: np.std(dataset.T, axis=x) for x in [None, 0, 1]
    })


dataset_1 = np.array([[1, 2], [3, 4]])

dataset_2 = np.array([[0, 0], [0, 0]])

dataset_3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0], [9, 10, 11,
                                                                 12]])
(expected_mean_for_dataset_1,
 expected_std_for_dataset_1) = _make_expected_targets_for_dataset(dataset_1)
(expected_mean_for_dataset_2,
 expected_std_for_dataset_2) = _make_expected_targets_for_dataset(dataset_2)
(expected_mean_for_dataset_3,
 expected_std_for_dataset_3) = _make_expected_targets_for_dataset(dataset_3)

dataset_with_diff_shapes = [[1, 2], [1, 2, 3]]
expected_mean_for_dataset_with_diff_shapes = {
    None: np.array([1.8]),
}
expected_std_for_dataset_with_diff_shapes = {
    None: np.array([np.std([1, 2, 1, 2, 3])])
}


@pytest.mark.parametrize(
    "dataset,expected_mean,expected_std,axis",
    [(dataset_with_diff_shapes, expected_mean_for_dataset_with_diff_shapes,
      expected_std_for_dataset_with_diff_shapes, [None]),
     (dataset_1, expected_mean_for_dataset_1, expected_std_for_dataset_1,
      [None, 0, 1]),
     (dataset_2, expected_mean_for_dataset_2, expected_std_for_dataset_2,
      [None, 0, 1]),
     (dataset_3, expected_mean_for_dataset_3, expected_std_for_dataset_3,
      [None, 0, 1])])
def test_mean_and_std(dataset, expected_mean, expected_std, axis):
    for a in axis:
        dataset_with_len = DatasetWithLen(dataset)
        mean, std = _compute_mean_and_std(dataset_with_len, a)
        print(mean, std)
        assert np.allclose(mean, expected_mean[a])
        assert np.allclose(std, expected_std[a])

        dataset_with_iter = DatasetWithIter(dataset)
        mean, std = _compute_mean_and_std(dataset_with_iter, a)
        assert np.allclose(mean, expected_mean[a])
        assert np.allclose(std, expected_std[a])


def _compute_mean_and_std(dataset, axis):
    return dataset_statistics.compute_mean_and_std(dataset,
                                                   'a',
                                                   axis,
                                                   max_iterations=100)
