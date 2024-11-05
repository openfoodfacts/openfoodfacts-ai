import logging
import random
import string

import datasets
from openfoodfacts.images import download_image

logger = logging.getLogger(__name__)


def format_annotation_results_from_hf(
    objects: dict, image_width: int, image_height: int
):
    """Format annotation results from a HF object detection dataset into Label
    Studio format."""
    annotation_results = []
    for i in range(len(objects["bbox"])):
        bbox = objects["bbox"][i]
        # category_id = objects["category_id"][i]
        category_name = objects["category_name"][i]
        # These are relative coordinates (between 0.0 and 1.0)
        y_min, x_min, y_max, x_max = bbox
        # Make sure the coordinates are within the image boundaries,
        # and convert them to percentages
        y_min = min(max(0, y_min), 1.0) * 100
        x_min = min(max(0, x_min), 1.0) * 100
        y_max = min(max(0, y_max), 1.0) * 100
        x_max = min(max(0, x_max), 1.0) * 100
        x = x_min
        y = y_min
        width = x_max - x_min
        height = y_max - y_min

        id_ = "".join(random.choices(string.ascii_letters + string.digits, k=10))
        annotation_results.append(
            {
                "id": id_,
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [category_name],
                },
            },
        )
    return annotation_results


def format_object_detection_sample_from_hf(hf_sample: dict, split: str) -> dict:
    hf_meta = hf_sample["meta"]
    objects = hf_sample["objects"]
    image_width = hf_sample["width"]
    image_height = hf_sample["height"]
    annotation_results = format_annotation_results_from_hf(
        objects, image_width, image_height
    )
    return {
        "data": {
            "image_id": hf_sample["image_id"],
            "image_url": hf_meta["image_url"],
            "batch": "null",
            "split": split,
            "meta": {
                "width": image_width,
                "height": image_height,
                "barcode": hf_meta["barcode"],
                "off_image_id": hf_meta["off_image_id"],
            },
        },
        "predictions": [{"result": annotation_results}],
    }


def format_object_detection_sample_to_ls(
    image_id: str,
    image_url: str,
    width: int,
    height: int,
    extra_meta: dict | None = None,
) -> dict:
    """Format an object detection sample in Label Studio format.

    Args:
        image_id: The image ID.
        image_url: The URL of the image.
        width: The width of the image.
        height: The height of the image.
        extra_meta: Extra metadata to include in the sample.
    """
    extra_meta = extra_meta or {}
    return {
        "data": {
            "image_id": image_id,
            "image_url": image_url,
            "batch": "null",
            "meta": {
                "width": width,
                "height": height,
                **extra_meta,
            },
        },
    }


def format_object_detection_sample_to_hf(
    task_data: dict, annotations: list[dict], category_names: list[str]
) -> dict | None:
    if len(annotations) > 1:
        logger.info("More than one annotation found, skipping")
        return None
    elif len(annotations) == 0:
        logger.info("No annotation found, skipping")
        return None

    annotation = annotations[0]
    bboxes = []
    bbox_category_ids = []
    bbox_category_names = []

    for annotation_result in annotation["result"]:
        if annotation_result["type"] != "rectanglelabels":
            raise ValueError("Invalid annotation type: %s" % annotation_result["type"])

        value = annotation_result["value"]
        x_min = value["x"] / 100
        y_min = value["y"] / 100
        width = value["width"] / 100
        height = value["height"] / 100
        x_max = x_min + width
        y_max = y_min + height
        bboxes.append([y_min, x_min, y_max, x_max])
        category_name = value["rectanglelabels"][0]
        bbox_category_names.append(category_name)
        bbox_category_ids.append(category_names.index(category_name))

    image_url = task_data["image_url"]
    image = download_image(image_url, error_raise=False)
    if image is None:
        logger.error("Failed to download image: %s", image_url)
        return None

    return {
        "image_id": task_data["image_id"],
        "image": image,
        "width": task_data["meta"]["width"],
        "height": task_data["meta"]["height"],
        "meta": {
            "barcode": task_data["meta"]["barcode"],
            "off_image_id": task_data["meta"]["off_image_id"],
            "image_url": image_url,
        },
        "objects": {
            "bbox": bboxes,
            "category_id": bbox_category_ids,
            "category_name": bbox_category_names,
        },
    }


# The HuggingFace Dataset features
HF_DS_FEATURES = datasets.Features(
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
