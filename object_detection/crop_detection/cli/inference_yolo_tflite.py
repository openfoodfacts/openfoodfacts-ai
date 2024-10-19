from typing import Tuple, Dict, Optional
import argparse
from PIL import Image
import logging

import numpy as np
from cv2 import dnn
import tflite_runtime.interpreter as tflite


MODEL_PATH = "models/yolov8n_float16.tflite"
DETECTION_THRESHOLD = 0.8
NMS_THRESHOLD = 0.45

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def parse():
    parser = argparse.ArgumentParser(
        description="""Detect product boundary box. 
    Return a dictionnary with the score and the box relative coordinates (between [0, 1]).
    If --save-path is indicated, save the cropped image to the specified path.
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the image to be processed.",
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=MODEL_PATH, 
        help="Path to the .tflite model."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DETECTION_THRESHOLD,
        help="Detection score threshold.",
    )
    parser.add_argument(
        "--nms-threshold", 
        type=float, 
        default=NMS_THRESHOLD,
        help="Non-Maximum Suppression threshold."
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set debug mode.",
    )
    parser.add_argument(
        "--save-path", 
        type=str, 
        required=False, 
        help="Path to save the cropped image."
    )
    return parser.parse_args()


def main():
    args = parse()
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)
    detected_object = run_inference(
        image_path=args.image_path,
        threshold=args.threshold,
        nms_threshold=args.nms_threshold,
        interpreter=tflite.Interpreter(model_path=args.model_path),
    )
    print(detected_object)
    if args.save_path:
        LOGGER.info(f"Saving cropped image to {args.save_path}")
        save_detected_object(
            image_path=args.image_path,
            detected_object=detected_object,
            output_path=args.save_path,
        )


def _preprocess_image(image: Image, input_shape: Tuple[int, int]) -> np.ndarray:
    """Preprocess the image to match the model input size."""
    max_size = max(image.size)
    squared_image = Image.new("RGB", (max_size, max_size), color="black")
    squared_image.paste(image, (0, 0))
    resized_squared_image = squared_image.resize(input_shape)
    # Normalize the pixel values (scale between 0 and 1 if needed)
    normalized_image = np.array(resized_squared_image) / 255.0
    # Expand dimensions to match the model input shape [1, height, width, 3]
    input_tensor = np.expand_dims(normalized_image, axis=0).astype(np.float32)
    return input_tensor


def run_inference(
    image_path: str,
    threshold: float,
    nms_threshold: float,
    interpreter: tflite.Interpreter,
) -> Dict:
    # Init Tensorflow Lite
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    LOGGER.debug(f"Input details: {input_details}")
    LOGGER.debug(f"Output details: {output_details}")

    # Get shape required by the model
    input_shape = input_details[0]["shape"][1:3].tolist()

    image = Image.open(image_path)
    width, height = image.size

    # We keep image ratio after resizing to 640
    image_ratio = width / height
    if image_ratio > 1:
        scale_x = input_shape[0]
        scale_y = input_shape[1] / image_ratio
    else:
        scale_x = input_shape[0] * image_ratio
        scale_y = input_shape[1]

    # Prepare image for Yolo
    input_tensor = _preprocess_image(image, input_shape)
    assert list(input_tensor.shape) == list(input_details[0]["shape"])
    LOGGER.debug(f"Input tensor shape: {input_tensor.shape}")

    # Inference
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details[0]["index"])
    LOGGER.debug(f"Output tensor shape: {output_tensor.shape}")
    result_tensor = np.squeeze(output_tensor, axis=0).T

    # Post-process the result
    boxes = result_tensor[:, :4]
    scores = result_tensor[:, 4]
    detected_objects = []
    for i, score in enumerate(scores):
        if score > threshold:
            # YOLOv8 typically outputs (cx, cy, w, h) normalized in the range [0, 1]
            cx, cy, w, h = boxes[i]

            # Convert to corner coordinates
            xmin = (cx - w / 2) * input_shape[0]
            ymin = (cy - h / 2) * input_shape[1]
            xmax = (cx + w / 2) * input_shape[0]
            ymax = (cy + h / 2) * input_shape[1]

            # Bounding box coordinates are normalized to the input image size
            scaled_xmin = max(0.0, min(1.0, xmin / scale_x))
            scaled_ymin = max(0.0, min(1.0, ymin / scale_y))
            scaled_xmax = max(0.0, min(1.0, xmax / scale_x))
            scaled_ymax = max(0.0, min(1.0, ymax / scale_y))

            detected_objects.append(
                {
                    "score": scores[i],
                    "box": [scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax],
                }
            )

    # Apply Non-Maximum Suppression to overlapping boxes
    raw_detected_boxes = [
        detected_object["box"] for detected_object in detected_objects
    ]
    raw_detected_scores = [
        detected_object["score"] for detected_object in detected_objects
    ]
    nms_indices = dnn.NMSBoxes(
        bboxes=raw_detected_boxes,
        scores=raw_detected_scores,
        score_threshold=threshold,
        nms_threshold=nms_threshold,
    )
    # We only look for one box per image
    detected_object = detected_objects[nms_indices[0]]

    return detected_object


def save_detected_object(
    image_path: str, detected_object: Dict, output_path: Optional[str] = None
) -> None:
    if not output_path:
        print(output_path)
        raise ValueError("Output path is required.")
    image = Image.open(image_path)
    width, height = image.size
    xmin, ymin, xmax, ymax = detected_object["box"]
    xmin = int(xmin * width)
    ymin = int(ymin * height)
    xmax = int(xmax * width)
    ymax = int(ymax * height)
    image = image.crop((xmin, ymin, xmax, ymax))
    image.save(output_path)
