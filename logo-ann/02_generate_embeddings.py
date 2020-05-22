import argparse
import pathlib
from typing import Iterable

import h5py
from more_itertools import chunked
import numpy as np
import tqdm

from efficientnet_pytorch import EfficientNet
import torch


def build_model(model_type: str):
    return EfficientNet.from_pretrained(model_type)


def get_output_dim(model_type: str):
    if model_type == "efficientnet-b0":
        return 1280

    raise ValueError("unknown model type: {}".format(model_type))


def get_item_count(file_path: pathlib.Path) -> int:
    with h5py.File(str(file_path), "r") as f:
        image_dset = f["image"]
        return len(image_dset)


def generate_embeddings_iter(
    model, file_path: pathlib.Path, batch_size: int, device, min_confidence: float = 0.5
):
    with h5py.File(str(file_path), "r") as f:
        image_dset = f["image"]
        barcode_dset = f["barcode"]
        image_id_dset = f["image_id"]
        resolution_dset = f["resolution"]
        bounding_box_dset = f["bounding_box"]
        confidence_dset = f["confidence"]

        for slicing in chunked(range(len(image_dset)), batch_size):
            slicing = np.array(slicing)
            mask = image_id_dset[slicing] == 0

            if np.all(mask):
                break

            mask = (~mask) & (confidence_dset[slicing] >= min_confidence)

            if np.all(~mask):
                continue

            images = image_dset[slicing][mask]
            images = np.moveaxis(images, -1, 1)  # move channel dim to 1st dim

            with torch.no_grad():
                torch_images = torch.tensor(images, dtype=torch.float32, device=device)
                embeddings = model.extract_features(torch_images).cpu().numpy()

            max_embeddings = np.max(embeddings, (-1, -2))
            yield (
                max_embeddings,
                barcode_dset[slicing][mask],
                image_id_dset[slicing][mask],
                resolution_dset[slicing][mask],
                bounding_box_dset[slicing][mask],
                confidence_dset[slicing][mask],
            )


def generate_embedding_from_hdf5(
    data_gen: Iterable, output_path: pathlib.Path, output_dim: int, count: int,
):
    with h5py.File(str(output_path), "w") as f:
        embedding_dset = f.create_dataset(
            "embedding", (count, output_dim), dtype="f", chunks=True
        )
        barcode_dset = f.create_dataset(
            "barcode", (count,), dtype=h5py.string_dtype(), chunks=True
        )
        image_id_dset = f.create_dataset("image_id", (count,), dtype="i", chunks=True)
        resolution_dset = f.create_dataset(
            "resolution", (count, 2), dtype="i", chunks=True
        )
        bounding_box_dset = f.create_dataset(
            "bounding_box", (count, 4), dtype="f", chunks=True
        )
        confidence_dset = f.create_dataset(
            "confidence", (count,), dtype="f", chunks=True
        )

        offset = 0
        for batch in data_gen:
            (
                embeddings_batch,
                barcode_batch,
                image_id_batch,
                resolution_batch,
                bounding_box_batch,
                confidence_batch,
            ) = batch
            slicing = slice(offset, offset + len(embeddings_batch))
            embedding_dset[slicing] = embeddings_batch
            barcode_dset[slicing] = barcode_batch
            image_id_dset[slicing] = image_id_batch
            resolution_dset[slicing] = resolution_batch
            bounding_box_dset[slicing] = bounding_box_batch
            confidence_dset[slicing] = confidence_batch
            offset += len(embeddings_batch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.data_path.is_file()
    assert not args.output_path.is_file()
    model_type = "efficientnet-b0"
    model = build_model(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    model = model.to(device)
    data_gen = tqdm.tqdm(
        generate_embeddings_iter(
            model, args.data_path, args.batch_size, device, args.min_confidence
        )
    )
    output_dim = get_output_dim(model_type)
    count = get_item_count(args.data_path)
    generate_embedding_from_hdf5(data_gen, args.output_path, output_dim, count)
