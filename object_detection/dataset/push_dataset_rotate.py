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
import functools
import io
import logging
import pickle
from pathlib import Path

import datasets
import imagehash
import numpy as np

# import requests
import supervision as sv
import tensorflow as tf
import tqdm
from openfoodfacts.images import download_image, generate_image_path
from openfoodfacts.utils import get_logger
from PIL import Image

logger = get_logger(level=logging.DEBUG)
# session = requests.Session()

# Feature dictionary to parse the TFRecord files
feature_dict = {
    "image/height": tf.io.FixedLenFeature((), tf.int64, default_value=1),
    "image/width": tf.io.FixedLenFeature((), tf.int64, default_value=1),
    "image/filename": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/source_id": tf.io.FixedLenFeature((), tf.string, default_value=""),
    "image/key/sha256": tf.io.FixedLenFeature((), tf.string, default_value=""),
    # The image is stored as a string
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


# The HuggingFace Dataset features
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


def find_image_orientation(
    original_image: Image.Image, target_image: Image.Image
) -> int:
    original_image_fingerprint = imagehash.phash(original_image)

    for rotate in [0, 90, 180, 270]:
        if rotate != 0:
            target_rotated_image = target_image.rotate(rotate, expand=True)
        else:
            target_rotated_image = target_image
        target_fingerprint = imagehash.phash(target_rotated_image)
        diff_fingerprint = abs(target_fingerprint - original_image_fingerprint)
        print(
            f"Rotate: {rotate}, fingerprint: {target_fingerprint}, original: {original_image_fingerprint}, diff: {diff_fingerprint}"
        )
        if target_fingerprint == original_image_fingerprint:
            print("Found the image orientation: %d", rotate)
            return rotate

    raise ValueError("Cannot find the image orientation")


def display_image_with_bounding_boxes(
    image: Image.Image,
    y_min: np.ndarray,
    x_min: np.ndarray,
    y_max: np.ndarray,
    x_max: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
) -> Image.Image:
    width, height = image.size
    # Transform y_min, x_min, y_max, x_max in a single array of shape
    # (n_boxes, 4) (x1, y1, x2, y2)
    boxes = np.stack(
        [x_min * width, y_min * height, x_max * width, y_max * height], axis=-1
    )
    detections = sv.Detections(
        boxes, class_id=label_ids, data={"class_name": label_names}
    )

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    annotated_frame = bounding_box_annotator.annotate(
        scene=image.copy(), detections=detections
    )
    return annotated_frame


def rotate_bounding_boxes(
    y_min: np.ndarray,
    x_min: np.ndarray,
    y_max: np.ndarray,
    x_max: np.ndarray,
    rotate: int,
) -> tuple[np.array, np.array, np.array, np.array]:
    """Rotate bounding boxes using a given angle (counter-clockwise).

    All coordinates are in relative coordinates (0-1).
    """
    if rotate == 90:
        # y_min = old_x_min
        # x_min = 1 - old_y_max
        # y_max = old_x_max
        # x_max = 1 - old_y_min
        y_min, x_min, y_max, x_max = x_min, 1 - y_max, x_max, 1 - y_min
    if rotate == 180:
        # y_min = 1 - old_y_max
        # x_min = 1 - old_x_max
        # y_max = 1 - old_y_min
        # x_max = 1 - old_x_min
        y_min, x_min, y_max, x_max = 1 - y_max, 1 - x_max, 1 - y_min, 1 - x_min
    if rotate == 270:
        # y_min = 1 - old_x_max
        # x_min = old_y_min
        # y_max = 1 - old_x_min
        # x_max = old_y_max
        y_min, x_min, y_max, x_max = 1 - x_max, y_min, 1 - x_min, y_max

    return y_min, x_min, y_max, x_max


def generate_samples(
    ds,
    category_names: list[str],
    use_filename: bool = False,
):
    for i, tf_record in enumerate(tqdm.tqdm(ds, desc="TFRecord")):
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

        # Debug
        if (barcode, off_image_id) != ("0049000027624", "2"):
            continue

        format = tf_record["image/format"].numpy().decode("utf-8")
        assert format == "jpeg", f"Invalid format: {format}"
        xmin = tf.sparse.to_dense(tf_record["image/object/bbox/xmin"]).numpy()
        xmax = tf.sparse.to_dense(tf_record["image/object/bbox/xmax"]).numpy()
        ymin = tf.sparse.to_dense(tf_record["image/object/bbox/ymin"]).numpy()
        ymax = tf.sparse.to_dense(tf_record["image/object/bbox/ymax"]).numpy()

        encoded = tf_record["image/encoded"].numpy()

        with io.BytesIO(encoded) as f:
            original_image = Image.open(f)
            original_image.load()

        category_text = [
            x.decode("utf-8")
            for x in tf.sparse.to_dense(tf_record["image/object/class/text"])
            .numpy()
            .tolist()
        ]
        category_ids = [category_names.index(x) for x in category_text]

        image_path = generate_image_path(barcode, off_image_id)
        image = download_image(
            (barcode, off_image_id),
            use_cache=True,
            error_raise=False,
            # session=session,
        )

        if image is None:
            continue

        rotation = find_image_orientation(original_image, image)

        if rotation != 0:
            # We need to rotate the bounding boxes
            ymin, xmin, ymax, xmax = rotate_bounding_boxes(
                ymin, xmin, ymax, xmax, rotation
            )

        # Debug: display image with bounding boxes
        display_image_with_bounding_boxes(
            image, ymin, xmin, ymax, xmax, np.array(category_ids), category_text
        ).save(f"/home/raphael/Desktop/{barcode}_{off_image_id}.jpg")

        image_url = f"https://images.openfoodfacts.org/images/products{image_path}"

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
        yield convert_sample_to_hf_format(item)


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


def pickle_generator(dir: Path):
    for pkl in dir.glob("*.pkl"):
        with open(pkl, "rb") as f:
            yield pickle.load(f)


def push_from_local_file(
    input_dir: Path,
    repo_id: str,
    category_names: list[str],
    use_filename: bool,
):
    """Convert Tensorflow TFRecord to HuggingFace Dataset.

    Args:
        input_dir (Path): the directory containing the TFRecord files
        repo_id (str): the HuggingFace repository ID
        category_names (list[str]): the category names
        use_filename (bool): if True, use the filename as image_id, otherwise
            use the source_id
    """
    for split in [
        "val",
        "train",
    ]:
        tf_record_paths = list(str(x) for x in input_dir.glob(f"{split}.record*"))
        raw_ds = tf.data.TFRecordDataset(tf_record_paths)
        ds = raw_ds.map(_parse_image_function)

        output_dir = Path(f"/home/raphael/Desktop/{repo_id}/{split}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(
            generate_samples(ds, category_names, use_filename=use_filename)
        ):
            # Save output as pickle
            with open(output_dir / f"{i:05}.pkl", "wb") as f:
                pickle.dump(sample, f)

        breakpoint()
        hf_ds = datasets.Dataset.from_generator(
            functools.partial(pickle_generator, output_dir),
            features=hf_ds_features,
        )
        hf_ds.push_to_hub(repo_id, split=split)


if __name__ == "__main__":
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
    )

    # push_from_local_file(
    #     Path("/home/raphael/datasets/universal-logo-detector/tfrecord"),
    #     "openfoodfacts/universal-logo-detector",
    #     [
    #         "brand",
    #         "label",
    #     ],
    #     use_filename=True,
    # )
