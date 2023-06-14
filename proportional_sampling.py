"""Proportional Sampling."""

##############################################################################
# Write a function to pick a number proportional to its absolute value
##############################################################################

import numpy as np


def proportional_sampling(data: np.array):
    """Proportional Sampling.

    Args:
        data (np.array): Data

    Returns:
        One of the element of data
    """
    data = np.sort(np.abs(data))
    prob = np.cumsum(data) / np.sum(data)
    rand = np.random.uniform(size=1)[0]
    idx = np.searchsorted(prob, rand, side="right")
    if idx >= data.size:
        idx = data.size - 1
    return data[idx]


if "__main__" == __name__:
    arr = np.array([-4, -1, 0, 9, 3, 6, 2])
    print(arr)

    test = dict()
    n_trails = 100000
    for _ in range(n_trails):
        number = proportional_sampling(arr)
        if number not in test.keys():
            test[number] = 1
        else:
            test[number] += 1

    print(test)
