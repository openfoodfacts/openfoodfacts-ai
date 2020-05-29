import argparse
import pathlib
from typing import Iterable, Set

import h5py
from more_itertools import chunked
import numpy as np
import tqdm

from efficientnet_pytorch import EfficientNet
import torch

from utils import get_offset, get_seen_set


def build_model(model_type: str):
    return EfficientNet.from_pretrained(model_type)


def get_output_dim(model_type: str):
    if model_type == "efficientnet-b0":
        return 1280

    if model_type == "efficientnet-b5":
        return 2048

    raise ValueError("unknown model type: {}".format(model_type))


def get_item_count(file_path: pathlib.Path) -> int:
    with h5py.File(str(file_path), "r") as f:
        image_dset = f["image"]
        return len(image_dset)


def generate_embeddings_iter(
    model,
    file_path: pathlib.Path,
    batch_size: int,
    device: torch.device,
    seen_set: Set[int],
    min_confidence: float = 0.5,
):
    with h5py.File(str(file_path), "r") as f:
        image_dset = f["image"]
        confidence_dset = f["confidence"]
        external_id_dset = f["external_id"]

        for slicing in chunked(range(len(image_dset)), batch_size):
            slicing = np.array(slicing)
            external_ids = external_id_dset[slicing]
            mask = external_ids == 0

            if np.all(mask):
                break

            mask = (~mask) & (confidence_dset[slicing] >= min_confidence)

            for i, external_id in enumerate(external_ids):
                if int(external_id) in seen_set:
                    mask[i] = 0

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
                external_ids[mask],
            )


def generate_embedding_from_hdf5(
    data_gen: Iterable, output_path: pathlib.Path, output_dim: int, count: int,
):
    file_exists = output_path.is_file()

    with h5py.File(str(output_path), "a") as f:
        if not file_exists:
            embedding_dset = f.create_dataset(
                "embedding", (count, output_dim), dtype="f", chunks=True
            )
            external_id_dset = f.create_dataset(
                "external_id", (count,), dtype="i", chunks=True
            )
            offset = 0
        else:
            offset = get_offset(f)
            embedding_dset = f["embedding"]
            external_id_dset = f["external_id"]

        print("Offset: {}".format(offset))

        for (embeddings_batch, external_id_batch) in data_gen:
            slicing = slice(offset, offset + len(embeddings_batch))
            embedding_dset[slicing] = embeddings_batch
            external_id_dset[slicing] = external_id_batch
            offset += len(embeddings_batch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--model-type", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.data_path.is_file()
    assert not args.output_path.is_file()
    assert (
        args.model_type.startswith("efficientnet-b") and args.model_type[-1].isdigit()
    )
    model_type = args.model_type
    model = build_model(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    model = model.to(device)

    seen_set = get_seen_set(args.output_path)
    print("Number of seen items: {}".format(len(seen_set)))

    data_gen = tqdm.tqdm(
        generate_embeddings_iter(
            model,
            args.data_path,
            args.batch_size,
            device,
            seen_set,
            args.min_confidence,
        )
    )
    output_dim = get_output_dim(model_type)
    count = get_item_count(args.data_path)
    generate_embedding_from_hdf5(data_gen, args.output_path, output_dim, count)
