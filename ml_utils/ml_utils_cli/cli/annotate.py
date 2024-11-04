import random
import string

from cli.triton.object_detection import ObjectDetectionResult
from openfoodfacts.utils import get_logger
from ultralytics.engine.results import Results

logger = get_logger(__name__)


def format_annotation_results_from_triton(
    objects: list[ObjectDetectionResult], image_width: int, image_height: int
):
    """Format annotation results from a Triton object detection model into
    Label Studio format."""
    annotation_results = []
    for object_ in objects:
        bbox = object_.bounding_box
        category_name = object_.label
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

        id_ = generate_id()
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


def format_annotation_results_from_ultralytics(
    results: Results,
    labels: list[str],
    label_mapping: dict[str, str] | None = None,
) -> list[dict]:
    annotation_results = []
    orig_height, orig_width = results.orig_shape
    boxes = results.boxes
    classes = boxes.cls.tolist()
    for i, xyxyn in enumerate(boxes.xyxyn):
        # Boxes found.
        if len(xyxyn) > 0:
            xyxyn = xyxyn.tolist()
            x1 = xyxyn[0] * 100
            y1 = xyxyn[1] * 100
            x2 = xyxyn[2] * 100
            y2 = xyxyn[3] * 100
            width = x2 - x1
            height = y2 - y1
            label_id = int(classes[i])
            label_name = labels[label_id]
            if label_mapping:
                label_name = label_mapping.get(label_name, label_name)
            annotation_results.append(
                {
                    "id": generate_id(),
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
                        "rectanglelabels": [label_name],
                    },
                },
            )
    return annotation_results


def generate_id(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))
