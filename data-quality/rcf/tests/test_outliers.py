import numpy as np
from rcf.outliers import detect_iqr_outliers


def test_detects_clear_outlier():
    
    data = [1, 2, 2, 3, 100]
    mask = detect_iqr_outliers(data)

    assert mask.tolist() == [False, False, False, False, True]


def test_no_outliers_in_normal_range():
    
    data = [10, 11, 12, 13, 14]
    mask = detect_iqr_outliers(data)

    assert not any(mask)


def test_handles_missing_values():
    
    data = [1, 2, None, 3, 100]
    mask = detect_iqr_outliers(data)

    assert bool(mask[-1]) is True


def test_returns_indices_when_requested():
    
    data = [5, 6, 7, 100]
    indices = detect_iqr_outliers(data, return_mask=False)

    assert indices.tolist() == [3]
