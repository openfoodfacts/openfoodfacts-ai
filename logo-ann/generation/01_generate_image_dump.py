import argparse
import json
from math import floor
import pathlib
import re
from typing import Iterable, List, Optional, Set, Tuple

import h5py
import numpy as np
import lycon
from PIL import Image
from more_itertools import chunked
import tqdm

from utils import get_offset, get_seen_set

"""Return a hdf5 file containing all the data about the logos of the OpenFoodFacts database.

> > > python3 01_generate_image_dump.py image_dir data_path output_path --size S (--compression) --count C

image_dir: path of the directory containing all the products image to crop
data_path: path of the jsonl file containing the logo_id, barcode, image_id, bounding_box and confidence score for each logo
output_path: path where the hdf5 file will be returned
size: size used to resize the logos. They are returned as squares of shape size * size. (for CLIP models the expected size is 224)
compression: format of the compression for the intern data of the hdf5 file
count: amount of logos you want to save in the hdf5 file. By default, it will save all the logos of the data file.

"""


def save_hdf5(
    output_file: pathlib.Path,
    data_iter: Iterable[
        Tuple[str, int, np.ndarray, Tuple[int, int], List[float], float, int]
    ],
    count: int,
    size: int,
    batch_size: int = 256,
    compression: Optional[str] = None,
):

    """Write all the outputs yielded by the get_data_gen function in an hdf5 file (create the file if it doesn't already exist).
    For that, create different h5py datasets for every data and add it in, batch after batch.
    """

    file_exists = output_file.is_file()

    with h5py.File(str(output_file), "a") as f:
        if not file_exists:
            image_dset = f.create_dataset(
                "image",
                (count, size, size, 3),
                dtype="uint8",
                compression=compression,
                chunks=(2048, size, size, 3),
            )
            barcode_dset = f.create_dataset(
                "barcode",
                (count,),
                dtype=h5py.string_dtype(),
                compression=compression,
                chunks=(2048,),
            )
            image_id_dset = f.create_dataset(
                "image_id", (count,), dtype="i", compression=compression, chunks=(2048,)
            )
            resolution_dset = f.create_dataset(
                "resolution",
                (count, 2),
                dtype="i",
                compression=compression,
                chunks=(2048, 2),
            )
            bounding_box_dset = f.create_dataset(
                "bounding_box",
                (count, 4),
                dtype="f",
                compression=compression,
                chunks=(2048, 4),
            )
            confidence_dset = f.create_dataset(
                "confidence",
                (count,),
                dtype="f",
                compression=compression,
                chunks=(2048,),
            )
            external_id_dset = f.create_dataset(
                "external_id",
                (count,),
                dtype="i",
                compression=compression,
                chunks=(2048,),
            )
            offset = 0
        else:
            offset = get_offset(f)
            image_dset = f["image"]
            barcode_dset = f["barcode"]
            image_id_dset = f["image_id"]
            resolution_dset = f["resolution"]
            bounding_box_dset = f["bounding_box"]
            confidence_dset = f["confidence"]
            external_id_dset = f["external_id"]

        print("Offset: {}".format(offset))

        for batch in chunked(
            data_iter, batch_size
        ):  # iter over data_iter through batches of batch_size elements of data_iter
            barcode_batch = np.array([x[0] for x in batch])
            image_id_batch = np.array([x[1] for x in batch], dtype="i")
            image_batch = np.stack([x[2] for x in batch], axis=0)
            resolution_batch = np.array([list(x[3]) for x in batch])
            bounding_box_batch = np.array([list(x[4]) for x in batch])
            confidence_batch = np.array([x[5] for x in batch])
            external_id_batch = np.array([x[6] for x in batch])
            slicing = slice(offset, offset + len(batch))
            barcode_dset[slicing] = barcode_batch
            image_id_dset[slicing] = image_id_batch
            image_dset[slicing] = image_batch
            resolution_dset[slicing] = resolution_batch
            bounding_box_dset[slicing] = bounding_box_batch
            confidence_dset[slicing] = confidence_batch
            external_id_dset[slicing] = external_id_batch
            offset += len(batch)


def crop_image(
    image: np.ndarray, bounding_box: Tuple[float, float, float, float]
) -> np.ndarray:

    """Return the cropped logo as an array extracted from the array of the image"""

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


def split_barcode(barcode: str) -> Optional[List[str]]:
    if not barcode.isdigit():
        print("unknown barcode format: {}".format(barcode))
        # raise ValueError("unknown barcode format: {}".format(barcode))
        return None

    match = BARCODE_PATH_REGEX.fullmatch(barcode)

    if match:
        return [x for x in match.groups() if x]

    return [barcode]


def generate_image_path(barcode: str, image_id: str) -> Optional[str]:
    splitted_barcode = split_barcode(barcode)
    if splitted_barcode is None:
        return None
    return "{}/{}.jpg".format("/".join(splitted_barcode), image_id)


def count_results(base_image_dir: pathlib.Path, result_path: pathlib.Path) -> int:
    count = 0
    for item in iter_jsonl(result_path):
        barcode = int(item["barcode"])
        image_id = int(item["image_id"])
        generated_image_path = generate_image_path(str(barcode), str(image_id))
        if generated_image_path == None:
            continue
        file_path = base_image_dir / generated_image_path

        if not file_path.is_file():
            continue

        results = item["result"]
        count += len(results)

    return count


def get_data_gen(
    base_image_dir: pathlib.Path, data_path: pathlib.Path, size: int, seen_set: Set[int]
) -> Iterable[Tuple[str, int, np.ndarray, Tuple[int, int], List[float], float, int]]:

    """Inputs:
    - base_image_dir: path of the directory containing the images from which to get logos data
    - data_path: path of the jsonl file containing annotation details about logos
    - size: size of the squared logos returned in the hdf5 file
    - seen_set: set of all logos already seen by the func allowing to run the file without starting from scratch if an error occured

    Yield the following outputs:
    - barcode: barcode of the product corresponding to the logo
    - image_id: id of the image from which the logo was extracted
    - cropped_resized_img: the resized logo
    - original_resolution: the original size of the image
    - bounding_box: an array of size 4 containing the coordinates of the square where the logo was detected
    - score: score of confidence regarding the validity of the cropped logo
    - logo_id: id of the logo
    """

    for logo_annotation in iter_jsonl(data_path):
        logo_id = logo_annotation["id"]

        if logo_id in seen_set:
            continue

        barcode = logo_annotation["barcode"]
        image_id = int(logo_annotation["image_id"])
        generated_image_path = generate_image_path(str(barcode), str(image_id))
        if generated_image_path == None:
            continue
        file_path = base_image_dir / generated_image_path

        if not file_path.is_file():
            continue

        base_img = lycon.load(str(file_path))
        if base_img is None:
            print("base_img is None :", base_img)
            continue

        if base_img.shape[-1] != 3:
            base_img = np.array(Image.fromarray(base_img).convert("RGB"))

        assert base_img.shape[-1] == 3

        bounding_box = logo_annotation["bounding_box"]
        score = logo_annotation["score"]
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
            logo_id,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=pathlib.Path)
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--compression")
    parser.add_argument("--count", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.image_dir.is_dir()
    assert args.data_path.is_file()
    assert args.size > 100
    print(
        "Generating image dump HDF5 file in {}, from data: {}, image dir: {}; size: {}".format(
            args.output_path, args.data_path, args.image_dir, args.size
        )
    )
    print("Number of items: {}".format(args.count))

    seen_set = get_seen_set(args.output_path)
    print("Number of seen items: {}".format(len(seen_set)))

    data_gen = tqdm.tqdm(
        get_data_gen(args.image_dir, args.data_path, args.size, seen_set)
    )
    save_hdf5(
        args.output_path, data_gen, args.count, args.size, compression=args.compression
    )
    print("Dump completed")
