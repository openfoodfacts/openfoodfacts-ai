"""This script converts object detection datasets from Tensorflow TFRecord
format to HuggingFace Dataset and Ultralytics format, and pushes the HF dataset
to HF Hub.

Previous object detection models were trained using Tensorflow Object Detection
API that require datasets to be stored in TFRecord format. This script fetches
datasets (either from a local directory or from a URL in case of a single-file
dataset) and converts them.

We provide 2 types of exports:
- HuggingFace Dataset: there are no standard format for HuggingFace Datasets,
    so we use a custom format. The dataset is pushed to HuggingFace Hub.
- Ultralytics: the dataset is converted to the Ultralytics format, and saved
    locally. The Ultralytics format is a directory containing:
    - images: the images
    - labels: the labels in a text file for each image
    - data.yaml: a YAML file containing the dataset configuration

As images in TF Serving size are not always the same as the original images
(they are often resized), we download the original images from the Open Food
Facts server (or from AWS S3 if the image is available there) and use them if
the image ratio is the same as the original image.
"""

import copy
import io
import math
from pathlib import Path
from typing import Optional, Union

import datasets
import numpy as np
import PIL
import requests
import tensorflow as tf
from openfoodfacts.images import generate_image_path
from openfoodfacts.utils import AssetLoadingException, get_asset_from_url
from PIL import Image

session = requests.Session()

feature_dict = {
    "image/height": tf.io.FixedLenFeature((), tf.int64, default_value=1),
    "image/width": tf.io.FixedLenFeature((), tf.int64, default_value=1),
    "image/filename": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/source_id": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/key/sha256": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/encoded": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/format": tf.io.FixedLenFeature((), tf.string, default_value="jpeg"),
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/class/text": tf.io.VarLenFeature(tf.string),
    "image/object/is_crowd": tf.io.VarLenFeature(tf.int64),
    "image/object/area": tf.io.VarLenFeature(tf.float32),
}


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_dict)


hf_ds_features = datasets.Features(
    {
        "image_id": datasets.Value("string"),
        "image": datasets.features.Image(),
        "width": datasets.Value("int64"),
        "height": datasets.Value("int64"),
        "meta": {
            "barcode": datasets.Value("string"),
            "off_image_id": datasets.Value("string"),
            "image_url": datasets.Value("string"),
        },
        "objects": {
            "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            "category_id": datasets.Sequence(datasets.Value("int64")),
            "category_name": datasets.Sequence(datasets.Value("string")),
        },
    }
)


def get_image_from_url(
    image_url: str, error_raise: bool = True, session: Optional[requests.Session] = None
) -> Union[tuple["Image.Image", bytes], tuple[None, None]]:
    """Fetch an image from `image_url` and load it.

    :param image_url: URL of the image to load
    :param error_raise: if True, raises a `AssetLoadingException` if an error
      occured, defaults to False. If False, None is returned if an error
      occured.
    :param session: requests Session to use, by default no session is used.
    :return: the Pillow Image or None.
    """
    r = get_asset_from_url(image_url, error_raise, session)
    if r is None:
        return None, None
    content_bytes = r.content

    try:
        return Image.open(io.BytesIO(content_bytes)), content_bytes
    except PIL.UnidentifiedImageError:
        error_message = f"Cannot identify image {image_url}"
        if error_raise:
            raise AssetLoadingException(error_message)
        print(error_message)
    except PIL.Image.DecompressionBombError:
        error_message = f"Decompression bomb error for image {image_url}"
        if error_raise:
            raise AssetLoadingException(error_message)
        print(error_message)

    return None, None


def get_full_image(
    image_path: str,
    width: int,
    height: int,
    raise_if_error: bool = True,
) -> Union[tuple[Image.Image, bytes], tuple[None, None]]:
    image_url = (
        f"https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/data{image_path}"
    )
    if requests.head(image_url).status_code != 200:
        # The image is not available on S3, try the static server
        print(f"Image not found on S3 ({image_url}), trying the static server")
        image_url = f"https://static.openfoodfacts.org/images/products{image_path}"

    full_image, image_bytes = get_image_from_url(
        image_url, session=session, error_raise=False
    )

    if full_image is None:
        error_message = (
            f"Image not found on the server ({image_url}), keeping the original image"
        )
        if raise_if_error:
            raise RuntimeError(error_message)
        print(error_message)
        return None, None

    ratio = width / height
    downloaded_image_ratio = full_image.width / full_image.height
    if not math.isclose(ratio, downloaded_image_ratio, abs_tol=1e-2):
        error_message = f"different ratio ({ratio} != {downloaded_image_ratio}), keeping the original image"
        if raise_if_error:
            raise RuntimeError(error_message)
        print(error_message)
        return None, None

    return full_image, image_bytes


def generate_samples(
    ds,
    category_names: list[str],
    split: str,
    use_filename: bool = False,
    raise_if_error: bool = True,
    image_dir: Optional[Path] = None,
):
    for tf_record in ds:
        height = tf_record["image/height"].numpy().item()
        width = tf_record["image/width"].numpy().item()
        if use_filename:
            filename = tf_record["image/filename"].numpy().decode("utf-8")
            image_id = filename.split(".")[0]
            barcode, off_image_id = image_id.split("_")
        else:
            source_id = tf_record["image/source_id"].numpy().decode("utf-8")
            image_id = source_id.split(".")[0]
            barcode, off_image_id = image_id.split("-")

        # key_sha256 = tf_record["image/key/sha256"].numpy().decode("utf-8")
        encoded = tf_record["image/encoded"].numpy()

        with io.BytesIO(encoded) as f:
            image = Image.open(f)
            image.load()

        format = tf_record["image/format"].numpy().decode("utf-8")
        assert format == "jpeg", f"Invalid format: {format}"
        xmin = tf.sparse.to_dense(tf_record["image/object/bbox/xmin"]).numpy()
        xmax = tf.sparse.to_dense(tf_record["image/object/bbox/xmax"]).numpy()
        ymin = tf.sparse.to_dense(tf_record["image/object/bbox/ymin"]).numpy()
        ymax = tf.sparse.to_dense(tf_record["image/object/bbox/ymax"]).numpy()

        category_text = [
            x.decode("utf-8")
            for x in tf.sparse.to_dense(tf_record["image/object/class/text"])
            .numpy()
            .tolist()
        ]
        category_ids = [category_names.index(x) for x in category_text]

        image_path = generate_image_path(barcode, off_image_id)
        image_url = f"https://static.openfoodfacts.org/images/products{image_path}"
        full_image, full_image_bytes = get_full_image(
            image_path, width, height, raise_if_error=raise_if_error
        )

        if full_image is not None:
            image = full_image
            image_bytes = full_image_bytes
        else:
            image_bytes = encoded

        if image_dir is not None:
            split_image_dir = image_dir / split
            split_image_dir.mkdir(parents=True, exist_ok=True)

            with (split_image_dir / f"{image_id}.jpg").open("wb") as f:
                f.write(image_bytes)

        width = image.width
        height = image.height
        item = {
            "image_id": image_id,
            "image": image,
            "width": width,
            "height": height,
            "meta": {
                "barcode": barcode,
                "off_image_id": off_image_id,
                "image_url": image_url,
            },
            "objects": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "category_id": category_ids,
                "category_name": category_text,
            },
        }
        yield item


def convert_sample_to_hf_format(sample: dict):
    sample = copy.deepcopy(sample)
    xmin = sample["objects"]["xmin"]
    ymin = sample["objects"]["ymin"]
    xmax = sample["objects"]["xmax"]
    ymax = sample["objects"]["ymax"]
    bboxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)
    return {
        "image_id": sample["image_id"],
        "image": sample["image"],
        "width": sample["width"],
        "height": sample["height"],
        "meta": {
            "barcode": sample["meta"]["barcode"],
            "off_image_id": sample["meta"]["off_image_id"],
            "image_url": sample["meta"]["image_url"],
        },
        "objects": {
            "bbox": bboxes,
            "category_id": sample["objects"]["category_id"],
            "category_name": sample["objects"]["category_name"],
        },
    }


def create_hf_dataset(repo_id: str, split: str, samples: list[dict]):
    hf_ds = datasets.Dataset.from_list(
        samples,
        features=hf_ds_features,
    )
    hf_ds.push_to_hub(repo_id, split=split)


def create_ultralytics_dataset(
    ultralytics_output_dir: Path,
    split: str,
    samples: list[dict],
    category_names: list[str],
):
    split_labels_dir = ultralytics_output_dir / "datasets" / "labels" / split
    split_labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        objects = sample["objects"]
        xmin = objects["xmin"]
        ymin = objects["ymin"]
        xmax = objects["xmax"]
        ymax = objects["ymax"]

        # Save the labels in the Ultralytics format:
        # - one label per line
        # - each line is a list of 5 elements:
        #   - category_id
        #   - x_center
        #   - y_center
        #   - width
        #   - height
        with (split_labels_dir / f"{sample['image_id']}.txt").open("w") as f:
            for category_id, (x1, y1, x2, y2) in zip(
                objects["category_id"], zip(xmin, ymin, xmax, ymax)
            ):
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

        with (ultralytics_output_dir / "data.yaml").open("w") as f:
            f.write("path: .\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test:\n")
            f.write("names:\n")
            for i, category_name in enumerate(category_names):
                f.write(f"  {i}: {category_name}\n")


def push_from_url(
    train_url: str,
    test_url: str,
    repo_id: str,
    category_names: list[str],
    use_filename: bool,
    raise_if_error: bool,
    create_hf: bool = True,
    create_ultralytics: bool = True,
    ultralytics_output_dir: Optional[Path] = None,
):
    """Convert Tensorflow TFRecord to HuggingFace Dataset.

    Args:
        train_url: URL to the training TFRecord file.
        test_url: URL to the testing TFRecord file.
    """
    if create_ultralytics:
        if not ultralytics_output_dir:
            raise ValueError("ultralytics_output_dir must be set")
        ultralytics_output_dir.mkdir(parents=True, exist_ok=True)

    for split, url in zip(["train", "val"], [train_url, test_url]):
        # Download the TFRecord files
        tfrecord_path = tf.keras.utils.get_file(origin=url)
        raw_ds = tf.data.TFRecordDataset(tfrecord_path)
        ds = raw_ds.map(_parse_image_function)
        samples = list(
            generate_samples(
                ds,
                category_names,
                split=split,
                use_filename=use_filename,
                raise_if_error=raise_if_error,
                image_dir=(
                    ultralytics_output_dir / "datasets" / "images"
                    if ultralytics_output_dir
                    else None
                ),
            )
        )
        if create_hf:
            create_hf_dataset(
                repo_id, split, [convert_sample_to_hf_format(s) for s in samples]
            )

        if create_ultralytics:
            create_ultralytics_dataset(
                ultralytics_output_dir, split, samples, category_names=category_names
            )


def push_from_local_file(
    input_dir: Path,
    repo_id: str,
    category_names: list[str],
    use_filename: bool,
    raise_if_error: bool,
    create_hf: bool = True,
    create_ultralytics: bool = True,
    ultralytics_output_dir: Optional[Path] = None,
):
    """Convert Tensorflow TFRecord to HuggingFace Dataset.

    Args:
        train_url: URL to the training TFRecord file.
        test_url: URL to the testing TFRecord file.
    """
    if create_ultralytics:
        if not ultralytics_output_dir:
            raise ValueError("ultralytics_output_dir must be set")
        ultralytics_output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        tf_record_paths = list(str(x) for x in input_dir.glob(f"{split}.record*"))
        raw_ds = tf.data.TFRecordDataset(tf_record_paths)
        ds = raw_ds.map(_parse_image_function)
        samples = list(
            generate_samples(
                ds,
                category_names,
                split=split,
                use_filename=use_filename,
                raise_if_error=raise_if_error,
                image_dir=(
                    ultralytics_output_dir / "datasets" / "images"
                    if ultralytics_output_dir
                    else None
                ),
            )
        )

        if create_hf:
            create_hf_dataset(
                repo_id, split, [convert_sample_to_hf_format(s) for s in samples]
            )

        if create_ultralytics:
            create_ultralytics_dataset(
                ultralytics_output_dir, split, samples, category_names=category_names
            )


if __name__ == "__main__":
    # push_from_url(
    #     ultralytics_output_dir=Path(
    #         "/home/raphael/datasets/nutriscore-detection/ultralytics"
    #     ),
    #     train_url="https://github.com/openfoodfacts/robotoff-models/releases/download/tf-nutriscore-1.0/train.record",
    #     test_url="https://github.com/openfoodfacts/robotoff-models/releases/download/tf-nutriscore-1.0/val.record",
    #     repo_id="openfoodfacts/nutriscore-object-detection",
    #     category_names=[
    #         "nutriscore-a",
    #         "nutriscore-b",
    #         "nutriscore-c",
    #         "nutriscore-d",
    #         "nutriscore-e",
    #     ],
    #     use_filename=False,
    #     raise_if_error=False,
    #     create_hf=True,
    #     create_ultralytics=False,
    # )

    push_from_local_file(
        Path("/home/raphael/datasets/nutrition-table-detection/tfrecord"),
        "openfoodfacts/nutrition-table-detection",
        [
            "nutrition-table",
            "nutrition-table-small",
            "nutrition-table-small-energy",
            "nutrition-table-text",
        ],
        use_filename=True,
        raise_if_error=False,
        create_hf=False,
        create_ultralytics=True,
        ultralytics_output_dir=Path(
            "/home/raphael/datasets/nutrition-table-detection/ultralytics"
        ),
    )
