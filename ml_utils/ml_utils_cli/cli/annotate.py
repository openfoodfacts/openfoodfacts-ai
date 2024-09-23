import os
import uuid
from typing import List, Iterable, Dict, Iterator
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

from openfoodfacts.utils import get_logger


logger = get_logger(__name__)

IMAGE_FORMAT = [".jpg", ".jpeg", ".png"]
MODEL_NAME = "models/yolov8x-worldv2.pt"
LABELS = ["packaging"]


def format_object_detection_sample_from_yolo(
    images_dir: Path,
    model_name: str,
    labels: List[str],
) -> Iterable[Dict]:
    logger.info("Loading images from %s", images_dir)
    image_paths = [image_path for image_path in images_dir.iterdir() if image_path.suffix in IMAGE_FORMAT]
    logger.info("Found %d images in %s", len(image_paths), images_dir)
    ls_data = generate_ls_data_from_images(image_paths=image_paths)
    logger.info("Pre-annotating images with YOLO")
    predictions = format_predictions_from_yolo(
        image_paths=image_paths,
        model_name=model_name,
        labels=labels
    )
    return [
        {
            "data": {
                "image_id": data["image_id"],
                "image_url": data["image_url"],
            },
            "predictions": [prediction] if prediction["result"] else [],
        }
        for data, prediction in zip(ls_data, predictions)
    ]


def generate_ls_data_from_images(image_paths: Iterable[Path]):
    for image_path in image_paths:
        yield {
            "image_id": image_path.stem.replace("_", "-"),
            "image_url": transform_id_to_url(image_path.name),
        }


def transform_id_to_url(image_id: str) -> str:
    """Format image_id: 325_938_117_1114_1 => https://images.openfoodfacts.org/images/products/325/938/117/1114/1"""
    off_base_url = "https://images.openfoodfacts.org/images/products/"
    return os.path.join(off_base_url, "/".join(image_id.split("_")))


def format_predictions_from_yolo(
    image_paths: Iterable[Path], 
    model_name: str,
    labels: List[str]
) -> Iterator[Dict]:
    results = pre_annotate_with_yolo(
        image_paths=image_paths,
        model_name=model_name,
        labels=labels,
    )
    for result in results:
        annotation_results = []
        orig_height, orig_width = result.orig_shape
        for xyxyn in result.boxes.xyxyn:
            # Boxes found.
            if len(xyxyn) > 0:
                xyxyn = xyxyn.tolist()
                x1 = xyxyn[0] * 100
                y1 = xyxyn[1] * 100
                x2 = xyxyn[2] * 100
                y2 = xyxyn[3] * 100
                width = x2 - x1
                height = y2 - y1
                annotation_results.append(
                    {
                        "id": str(uuid.uuid4())[:5],
                        "type": "rectanglelabels",
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": orig_width,
                        "original_height": orig_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": x1,
                            "y": y1,
                            "width": width,
                            "height": height,
                            "rectanglelabels": ["product"], # Label studio label
                        },
                    },
                )
        yield {
            "model_version": model_name,
            "result": annotation_results
        }


def pre_annotate_with_yolo(
    image_paths: Iterable[Path],
    model_name: str,
    labels: List[str],
    conf: float = 0.1,
    max_det: int = 1,
    stream: bool = True,
) -> Iterable[Results]:
    """https://docs.ultralytics.com/modes/predict/#working-with-results"""
    model = YOLO(model_name)
    model.set_classes(labels)
    results = model.predict(
        image_paths, 
        conf=conf, 
        max_det=max_det, 
        stream=stream,
    )
    return results
