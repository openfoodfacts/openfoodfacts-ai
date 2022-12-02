from collections import Counter
from main import (
    compute_metrics_k,
    get_distance_matrix_mask,
    pairwise_squared_euclidian_distance_numpy,
)
import numpy as np
import pytest


def pairwise_squared_euclidian_distance_naive(A):
    n = A.shape[0]
    output = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x = np.sum((A[j] - A[i]) ** 2)
            output[i, j] = x
            output[j, i] = x
    return output


def test_pairwise_squared_euclidian_distance_numpy():
    A = np.random.rand(32, 1024)
    assert np.isclose(
        pairwise_squared_euclidian_distance_numpy(A),
        pairwise_squared_euclidian_distance_naive(A),
        atol=1e-4,
    ).all()


@pytest.mark.parametrize("max_label_count", [0, 5, 10])
def test_get_distance_matrix_mask(max_label_count: int):
    size = 1000
    labels = np.random.randint(0, 10, size=size)
    output = get_distance_matrix_mask(size, max_label_count, labels)
    assert (np.diag(output) == 1).all()
    output[np.arange(size), np.arange(size)] = False
    if max_label_count == 0:
        assert (np.zeros((size, size), dtype=bool) == output).all()
    else:
        for row_idx in range(size):
            row = output[row_idx]
            selected_labels = labels[row]
            counter = Counter(map(int, selected_labels))
            for k, v in counter.items():
                assert v == max_label_count, f"{v} elements for key {k} (row {row_idx})"


@pytest.mark.parametrize(
    "distance_matrix,mask,labels,k_list,expected",
    [
        (
            np.array(
                [
                    [0, 5, 3, 4, 2],
                    [3, 4, 1, 5, 0],
                    [5, 4, 3, 2, 0],  # 2 pos/2 for label 2
                    [1, 0, 4, 5, 3],  # 0 pos/2 for label 2
                    [2, 3, 0, 1, 4],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [True, False, False, False, False],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, True, False, False],
                ],
                dtype=bool,
            ),
            np.array([0, 1, 2, 2, 3], dtype=int),
            [2],
            {
                "micro_precision": {2: np.array([0.0, 0.0, 0.5, 0.0])},
                "macro_precision": {2: np.array([0.0, 0.0, 0.75, 0.0])},
                "micro_recall": {2: np.array([0.0, 0.0, 0.75, 0.0])},
                "macro_recall": {2: np.array([0.0, 0.0, 0.75, 0.0])},
            },
        )
    ],
)
def test_compute_metrics_k(distance_matrix, mask, labels, k_list, expected):
    output = compute_metrics_k(distance_matrix, mask, labels, k_list)
    for key in list(output):
        output[key] = dict(output[key])
    print("\n", output)


# 'micro_precision': {2: array([0.  , 0.  , 0.75, 0.  ])}
# 'macro_precision': {2: array([0., 0., 0.75., 0.])}
# 'micro_recall': {2: array([0.  , 0.  , 0.75, 0.  ])}
# 'macro_recall': {2: array([0. , 0. , 1.5, 0. ])}
