import h5py
from utils import get_offset
import numpy as np

id = 11

with h5py.File("datasets/test-val_dataset.hdf5", 'r') as f:
    offset = get_offset(f)
    new_ids = f['external_id'][:offset]
    new_ids = np.array(new_ids)
    new_classes = f['class'][:offset]
    new_classes = np.array(new_classes)

breakpoint()