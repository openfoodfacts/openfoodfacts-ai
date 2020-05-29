import pathlib
from typing import Set

import h5py
import numpy as np


def get_offset(f: h5py.File) -> int:
    external_id_dset = f["external_id"]
    array = external_id_dset[:]
    non_zero_indexes = np.flatnonzero(array)
    return int(non_zero_indexes[-1]) + 1


def get_seen_set(hdf5_path: pathlib.Path) -> Set[int]:
    if hdf5_path.is_file():
        with h5py.File(str(hdf5_path), "r") as f:
            external_id_dset = f["external_id"]
            array = external_id_dset[:]
            non_zero_indexes = np.flatnonzero(array)
            max_offset = non_zero_indexes[-1]
            return set(int(x) for x in array[: max_offset + 1])

    return set()
