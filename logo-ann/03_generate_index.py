import argparse
import json
import pathlib

from annoy import AnnoyIndex
import h5py
from more_itertools import chunked
import numpy as np
import tqdm


def generate_embeddings_iter(file_path: pathlib.Path, batch_size: int):
    with h5py.File(str(file_path), "r") as f:
        embedding_dset = f["embedding"]
        barcode_dset = f["barcode"]
        image_id_dset = f["image_id"]
        resolution_dset = f["resolution"]
        bounding_box_dset = f["bounding_box"]
        confidence_dset = f["confidence"]

        for slicing in chunked(range(len(embedding_dset)), batch_size):
            slicing = np.array(slicing)
            mask = image_id_dset[slicing] == 0

            if np.all(mask):
                break

            mask = ~mask
            yield (
                embedding_dset[slicing][mask],
                barcode_dset[slicing][mask],
                image_id_dset[slicing][mask],
                resolution_dset[slicing][mask],
                bounding_box_dset[slicing][mask],
                confidence_dset[slicing][mask],
            )


def generate_index_from_hdf5(
    file_path: pathlib.Path,
    output_path: pathlib.Path,
    batch_size: int,
    tree_count: int = 100,
):
    data_gen = generate_embeddings_iter(file_path, batch_size)
    index = None
    offset: int = 0
    keys = []

    for batch in tqdm.tqdm(data_gen):
        (
            embeddings_batch,
            barcode_batch,
            image_id_batch,
            resolution_batch,
            bounding_box_batch,
            confidence_batch,
        ) = batch

        for (embedding, barcode, image_id, resolution, bounding_box, confidence) in zip(
            embeddings_batch,
            barcode_batch,
            image_id_batch,
            resolution_batch,
            bounding_box_batch,
            confidence_batch,
        ):
            if index is None:
                output_dim = embeddings_batch.shape[-1]
                index = AnnoyIndex(output_dim, "euclidean")

            index.add_item(offset, embedding)
            keys.append(
                {
                    "barcode": str(barcode),
                    "image_id": str(image_id),
                    "resolution": resolution.tolist(),
                    "bounding_box": bounding_box.tolist(),
                    "confidence": float(confidence),
                }
            )
            offset += 1

    if index is not None:
        index.build(tree_count)
        index.save(str(output_path))

        with output_path.with_suffix(".jsonl").open("w") as f:
            for key in keys:
                f.write(json.dumps(key) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--tree-count", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.data_path.is_file()
    assert not args.output_path.is_file()
    generate_index_from_hdf5(
        args.data_path, args.output_path, args.batch_size, args.tree_count,
    )
