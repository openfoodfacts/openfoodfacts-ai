import h5py
from utils import get_offset, get_labels
import settings
import numpy as np
import os
import re
import tqdm

def modify_hdf5_dataset(hdf5_file: str, changed_logos: dict):
    changed_ids = set(changed_logos.keys())

    with h5py.File(hdf5_file, 'a') as f:
        offset = get_offset(f)
        ids = f['external_id'][:offset]
        for i in range(len(ids)):
            id = str(ids[i])
            if id in changed_ids:
                f['class'][i] = changed_logos[id]

def get_changed_logos(missed_logos_dir: str):
    classes_str, classes_ids = get_labels(settings.labels_path, [])
    classes_str = np.array(classes_str)
    classes_ids = np.array(classes_ids)

    missed_logos = {}
    for root, dir, files in os.walk(missed_logos_dir):
        splitted = root.split('/')
        if splitted[-1] == 'true':
            predicted = splitted[-3]
            for logo in files:
                logo_id = re.search('\_.*\.', logo).group(0)[1:-1]
                missed_logos[logo_id] = int(classes_ids[np.where(classes_str==predicted)][0])
            true = splitted[-2]
    return missed_logos

if __name__ == '__main__':
    '''
    You can run this script once you have annotated each logo.
    * dataset_to_modify: paht to the hdf5 dataset you want to modify by your annotations.  
    * annotation_dir: directory you used to annotate logos.
    '''
    dataset_to_modify = "datasets/test-val_dataset.hdf5"
    annotation_dir = "missed_logos"
    changed_logos = get_changed_logos(annotation_dir)
    modify_hdf5_dataset("datasets/test-val_dataset.hdf5", changed_logos)