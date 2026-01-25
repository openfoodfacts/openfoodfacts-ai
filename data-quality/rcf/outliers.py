import numpy as np
import pandas as pd
from typing import Iterable, Union


NumericValues = Union[Iterable[float], np.ndarray, pd.Series]


def detect_iqr_outliers(
    values: NumericValues,
    multiplier: float = 1.5,
    return_mask: bool = True,
):


    series = pd.Series(values, dtype="float64")


    clean_series = series.dropna()


    if clean_series.empty:
        return (
            np.zeros(len(series), dtype=bool)
            if return_mask
            else np.array([], dtype=int)
        )

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1


    if iqr == 0:
        return (
            np.zeros(len(series), dtype=bool)
            if return_mask
            else np.array([], dtype=int)
        )

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (series < lower_bound) | (series > upper_bound)

    if return_mask:
        return outlier_mask.to_numpy()
    else:
        return np.where(outlier_mask)[0]
