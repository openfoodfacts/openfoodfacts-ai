import argparse
import pathlib

import faiss 
from annoy import AnnoyIndex
import h5py
from more_itertools import chunked
import numpy as np
import tqdm
import psutil

"""Return the files containing index and keys from an hdf5 file of embeddings. Depending on the KNN
library used, the number of files returned may vary.

> > >  python3 03_generation_index.py data_path output_path (--batch-size n) (--tree-count m) (--KNN-library str)

data_path: path of the hdf5 file containing the embeddings
output_path: path of the output file. You must only precise a name for the main file. If a key file has
            to be created, it will be done automatically.
batch-size: size of each batch of embeddings indexed unziped at the same time
KNN-library: name of the specific KNN library used (annoy, faiss)
"""

def generate_embeddings_iter(file_path: pathlib.Path, batch_size: int):
    with h5py.File(str(file_path), "r") as f:
        embedding_dset = f["embedding"]
        external_id_dset = f["external_id"]

        for slicing in chunked(range(len(embedding_dset)), batch_size):
            slicing = np.array(slicing)
            mask = external_id_dset[slicing] == 0

            if np.all(mask):
                break

            mask = ~mask
            yield (
                embedding_dset[slicing][mask],
                external_id_dset[slicing][mask],
            )


def generate_index_from_hdf5(
    file_path: pathlib.Path,
    output_path: pathlib.Path,
    batch_size: int,
    tree_count: int = 100,
):
    data_gen = generate_embeddings_iter(file_path, batch_size)
    index = None

    output_path = pathlib.Path(str(output_path)+"_faiss")
    assert not output_path.is_file()

    for batch in tqdm.tqdm(data_gen):
        (embeddings_batch, external_id_batch) = batch

        for (embedding, external_id) in zip(embeddings_batch, external_id_batch):
            if index is None:
                output_dim = embeddings_batch.shape[-1]
                index = faiss.index_factory(output_dim,"IDMap,HNSW")  # "IDMap,Flat" "IDMap,HNSW"

            index.add_with_ids(np.array([embedding]).astype('float32'),np.array([int(external_id)]))

    if index is not None:
        print("virtual memory",psutil.virtual_memory())
        faiss.write_index(index, str(output_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tree-count", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.data_path.is_file()
    generate_index_from_hdf5(
        args.data_path, args.output_path, args.batch_size, args.tree_count,
    )
