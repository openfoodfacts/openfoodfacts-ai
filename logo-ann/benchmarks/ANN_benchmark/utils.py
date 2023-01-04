import json
import numpy as np
import h5py
import pathlib
from more_itertools import chunked

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_embedding(embeddings_path: pathlib.Path, batch_size: int, nb_embeddings: int):
    """
    Get embeddings from an embeddings file.

    Parameters
    ----------
    embeddings_path : pathlib.Path
        Path to the embeddings file.
    batch_size : int
        Number of embeddings to return at once.
    nb_embeddings : int
        Maximum number of embeddings to return.

    Yields
    ------
    embeddings : numpy.ndarray
        Array of embeddings.
    external_ids : numpy.ndarray
        Array of external ids.
    """

    with h5py.File(str(embeddings_path), "r") as f:
        embedding_dset = f["embedding"]
        external_id_dset = f["external_id"]

        for slicing in chunked(range(min(len(embedding_dset), nb_embeddings)), batch_size):
            slicing = np.array(slicing)
            mask = external_id_dset[slicing] == 0

            if np.all(mask):
                break

            mask = ~mask
            yield (
                embedding_dset[slicing][mask],
                external_id_dset[slicing][mask],
            )

def save_data(file_name: str, data: dict):
    with open(file_name, 'w') as f:
        json.dump(data,f,cls=NumpyArrayEncoder)

def load_data(file_name: str):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

