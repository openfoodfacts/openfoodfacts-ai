import argparse
import json
from math import floor
import pathlib
import re
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np
import lycon
from PIL import Image
from more_itertools import chunked
import tqdm


def save_hdf5(
    output_file: pathlib.Path,
    data_iter: Iterable[
        Tuple[str, int, np.ndarray, Tuple[int, int], List[float], float]
    ],
    count: int,
    size: int,
    batch_size: int = 256,
    compression: Optional[str] = None,
    chunks: Optional[bool] = None,
):
    kwargs = {"chunks": chunks} if chunks else {}
    with h5py.File(str(output_file), "w") as f:
        image_dset = f.create_dataset(
            "image",
            (count, size, size, 3),
            dtype="uint8",
            compression=compression,
            **kwargs
        )
        barcode_dset = f.create_dataset(
            "barcode",
            (count,),
            dtype=h5py.string_dtype(),
            compression=compression,
            **kwargs
        )
        image_id_dset = f.create_dataset(
            "image_id", (count,), dtype="i", compression=compression, **kwargs
        )
        resolution_dset = f.create_dataset(
            "resolution", (count, 2), dtype="i", compression=compression, **kwargs
        )
        bounding_box_dset = f.create_dataset(
            "bounding_box", (count, 4), dtype="f", compression=compression, **kwargs
        )
        confidence_dset = f.create_dataset(
            "confidence", (count,), dtype="f", compression=compression, **kwargs
        )

        offset = 0

        for batch in chunked(data_iter, batch_size):
            barcode_batch = np.array([x[0] for x in batch])
            image_id_batch = np.array([x[1] for x in batch], dtype="i")
            image_batch = np.stack([x[2] for x in batch], axis=0)
            resolution_batch = np.array([list(x[3]) for x in batch])
            bounding_box_batch = np.array([list(x[4]) for x in batch])
            confidence_batch = np.array([x[5] for x in batch])
            slicing = slice(offset, offset + len(batch))
            barcode_dset[slicing] = barcode_batch
            image_id_dset[slicing] = image_id_batch
            image_dset[slicing] = image_batch
            resolution_dset[slicing] = resolution_batch
            bounding_box_dset[slicing] = bounding_box_batch
            confidence_dset[slicing] = confidence_batch
            offset += len(batch)


def crop_image(
    image: np.ndarray, bounding_box: Tuple[float, float, float, float]
) -> np.ndarray:
    ymin, xmin, ymax, xmax = bounding_box
    height, width = image.shape[:2]
    (left, right, top, bottom) = (
        floor(xmin * width),
        floor(xmax * width),
        floor(ymin * height),
        floor(ymax * height),
    )
    return image[top:bottom, left:right]


def iter_jsonl(path: pathlib.Path):
    with path.open("r") as f:
        for line in f:
            yield json.loads(line)


BARCODE_PATH_REGEX = re.compile(r"^(...)(...)(...)(.*)$")


def split_barcode(barcode: str) -> List[str]:
    if not barcode.isdigit():
        raise ValueError("unknown barcode format: {}".format(barcode))

    match = BARCODE_PATH_REGEX.fullmatch(barcode)

    if match:
        return [x for x in match.groups() if x]

    return [barcode]


def generate_image_path(barcode: str, image_id: str) -> str:
    splitted_barcode = split_barcode(barcode)
    return "{}/{}.jpg".format("/".join(splitted_barcode), image_id)


def count_results(base_image_dir: pathlib.Path, result_path: pathlib.Path) -> int:
    count = 0
    for item in iter_jsonl(result_path):
        barcode = int(item["barcode"])
        image_id = int(item["image_id"])
        file_path = base_image_dir / generate_image_path(str(barcode), str(image_id))

        if not file_path.is_file():
            continue

        results = item["result"]
        count += len(results)

    return count


def get_data_gen(
    base_image_dir: pathlib.Path, result_path: pathlib.Path, size: int
) -> Iterable[Tuple[str, int, np.ndarray, Tuple[int, int], List[float], float]]:
    for item in iter_jsonl(result_path):
        results = item["result"]

        if not results:
            continue

        barcode = item["barcode"]
        image_id = int(item["image_id"])
        file_path = base_image_dir / generate_image_path(barcode, str(image_id))

        if not file_path.is_file():
            continue

        base_img = lycon.load(str(file_path))
        assert base_img is not None

        if base_img.shape[-1] != 3:
            base_img = np.array(Image.fromarray(base_img).convert("RGB"))

        assert base_img.shape[-1] == 3

        for result in item["result"]:
            bounding_box = result["bounding_box"]
            score = result["score"]
            cropped_img = crop_image(base_img, bounding_box)
            original_height = int(cropped_img.shape[0])
            original_width = int(cropped_img.shape[1])
            original_resolution = (original_width, original_height)
            cropped_resized_img = lycon.resize(
                cropped_img,
                width=size,
                height=size,
                interpolation=lycon.Interpolation.CUBIC,
            )
            yield (
                barcode,
                image_id,
                cropped_resized_img,
                original_resolution,
                bounding_box,
                score,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=pathlib.Path)
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--chunks", action="store_true", default=False)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--compression")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.image_dir.is_dir()
    assert args.data_path.is_file()
    assert args.size > 100
    assert not args.output_path.exists(), "output file already exists: {}".format(
        args.output_path
    )
    print(
        "Generating image dump HDF5 file in {}, from data: {}, image dir: {}; size: {}, chunks: {}".format(
            args.output_path, args.data_path, args.image_dir, args.size, args.chunks
        )
    )
    # count = count_results(args.image_dir, args.data_path)
    count = 800000
    print("Number of items: {}".format(count))
    data_gen = tqdm.tqdm(get_data_gen(args.image_dir, args.data_path, args.size))
    save_hdf5(
        args.output_path,
        data_gen,
        count,
        args.size,
        compression=args.compression,
        chunks=args.chunks,
    )
    print("Dump completed")
