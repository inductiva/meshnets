"""File with functions to compute data statistics"""
import numpy as np


class OnlineStatistics:
    """Computes mean and standard deviation from streaming data

    Args:
      axis: Axis over which to compute. Default is None, which means
        that the statistics are computed over the flattened array.

    Example:

    >>> dataset = [[1, 2], [3, 4]]
    >>> stats = OnlineStatistics()
    >>> for data in dataset:
    ...     stats.update(data)
    >>> stats.get_mean()
    array([[2.5]])
    >>> stats.get_std()
    array([[1.11803399]])
    >>> stats = OnlineStatistics(axis=0)
    >>> for data in dataset:
    ...     stats.update(data)
    >>> stats.get_mean()
    array([[2., 3.]])
    >>> stats.get_std()
    array([[1., 1.]])
    >>> stats = OnlineStatistics(axis=1)
    >>> for data in dataset:
    ...     stats.update(data)
    >>> stats.get_mean()
    array([[1.5], [3.5]])
    >>> stats.get_std()
    array([[0.5], [0.5]])

    """

    def __init__(self, axis=None):
        self.axis = axis

        self.n = None
        self.cumulative_sum = None

        self.cumulative_sum_squared = None

    def update(self, example):
        data_point = np.expand_dims(example, 0)
        sum_element = np.sum(data_point, axis=self.axis, keepdims=True)
        sum_element_squared = np.sum(data_point**2,
                                     axis=self.axis,
                                     keepdims=True)
        if self.cumulative_sum is None:
            self.cumulative_sum = sum_element
            self.cumulative_sum_squared = sum_element_squared
        else:
            self.cumulative_sum = np.sum(np.concatenate(
                (self.cumulative_sum, sum_element)),
                                         axis=self.axis,
                                         keepdims=True)
            self.cumulative_sum_squared = np.sum(np.concatenate(
                (self.cumulative_sum_squared, sum_element_squared)),
                                                 axis=self.axis,
                                                 keepdims=True)
        n = np.sum(np.ones_like(data_point), axis=self.axis, keepdims=True)
        if self.n is None:
            self.n = n
        else:
            self.n = np.sum(np.concatenate((self.n, n)),
                            axis=self.axis,
                            keepdims=True)

    def get_mean(self):
        return self.cumulative_sum / self.n

    def get_std(self):
        return np.sqrt(self.cumulative_sum_squared * self.n -
                       self.cumulative_sum**2) / self.n
