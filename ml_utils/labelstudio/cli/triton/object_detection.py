import dataclasses
import functools
import logging
import time
from typing import Any, Optional

import grpc
import numpy as np
from PIL import Image
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.grpc.service_pb2_grpc import GRPCInferenceServiceStub

logger = logging.getLogger(__name__)


JSONType = dict[str, Any]

OBJECT_DETECTION_MODEL_VERSION = {
    "nutriscore": "tf-nutriscore-1.0",
    "nutrition_table": "tf-nutrition-table-1.0",
    "universal_logo_detector": "tf-universal-logo-detector-1.0",
}

LABELS = {
    "nutriscore": [
        "NULL",
        "nutriscore-a",
        "nutriscore-b",
        "nutriscore-c",
        "nutriscore-d",
        "nutriscore-e",
    ],
}

OBJECT_DETECTION_IMAGE_MAX_SIZE = (1024, 1024)


@functools.cache
def get_triton_inference_stub(
    triton_uri: str,
) -> GRPCInferenceServiceStub:
    """Return a gRPC stub for Triton Inference Server.

    :param triton_uri: URI of the Triton Inference Server
    :return: gRPC stub for Triton Inference Server
    """
    triton_uri = triton_uri
    channel = grpc.insecure_channel(triton_uri)
    return service_pb2_grpc.GRPCInferenceServiceStub(channel)


def convert_image_to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image into a numpy array.

    The image is converted to RGB if needed before generating the array.

    :param image: the input image
    :return: the generated numpy array of shape (width, height, 3)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


@dataclasses.dataclass
class ObjectDetectionResult:
    bounding_box: tuple[int, int, int, int]
    score: float
    label: str


@dataclasses.dataclass
class ObjectDetectionRawResult:
    num_detections: int
    detection_boxes: np.ndarray
    detection_scores: np.ndarray
    detection_classes: np.ndarray
    label_names: list[str]
    detection_masks: Optional[np.ndarray] = None
    boxed_image: Optional[Image.Image] = None

    def select(self, threshold: Optional[float] = None) -> list[ObjectDetectionResult]:
        if threshold is None:
            threshold = 0.5

        box_masks = self.detection_scores > threshold
        selected_boxes = self.detection_boxes[box_masks]
        selected_scores = self.detection_scores[box_masks]
        selected_classes = self.detection_classes[box_masks]

        results = []
        for bounding_box, score, label in zip(
            selected_boxes, selected_scores, selected_classes
        ):
            label_int = int(label)
            label_str = self.label_names[label_int]
            if label_str is not None:
                result = ObjectDetectionResult(
                    bounding_box=tuple(bounding_box.tolist()),  # type: ignore
                    score=float(score),
                    label=label_str,
                )
                results.append(result)

        return results

    def to_json(self, threshold: Optional[float] = None) -> list[JSONType]:
        return [dataclasses.asdict(r) for r in self.select(threshold)]


def resize_image(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    max_width, max_height = max_size

    if width > max_width or height > max_height:
        new_image = image.copy()
        new_image.thumbnail((max_width, max_height))
        return new_image

    return image


class RemoteModel:
    def __init__(self, name: str, label_names: list[str]):
        self.name: str = name
        self.label_names = label_names

    def detect_from_image(
        self,
        image: Image.Image,
        triton_uri: str,
    ) -> ObjectDetectionRawResult:
        """Run object detection model on an image.

        :param image: the input Pillow image
        :param triton_uri: URI of the Triton Inference Server.
        :return: the detection result
        """
        resized_image = resize_image(image, OBJECT_DETECTION_IMAGE_MAX_SIZE)
        image_array = convert_image_to_array(resized_image)
        grpc_stub = get_triton_inference_stub(triton_uri)
        request = service_pb2.ModelInferRequest()
        request.model_name = self.name

        image_input = service_pb2.ModelInferRequest().InferInputTensor()
        image_input.name = "inputs"
        image_input.datatype = "UINT8"
        image_input.shape.extend([1, image_array.shape[0], image_array.shape[1], 3])
        request.inputs.extend([image_input])

        for output_name in (
            "num_detections",
            "detection_classes",
            "detection_scores",
            "detection_boxes",
        ):
            output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            output.name = output_name
            request.outputs.extend([output])

        request.raw_input_contents.extend([image_array.tobytes()])
        start_time = time.monotonic()
        response = grpc_stub.ModelInfer(request)
        logger.debug(
            "Inference time for %s: %s", self.name, time.monotonic() - start_time
        )

        if len(response.outputs) != 4:
            raise Exception(f"expected 4 output, got {len(response.outputs)}")

        if len(response.raw_output_contents) != 4:
            raise Exception(
                f"expected 4 raw output content, got {len(response.raw_output_contents)}"
            )

        output_index = {output.name: i for i, output in enumerate(response.outputs)}
        num_detections = (
            np.frombuffer(
                response.raw_output_contents[output_index["num_detections"]],
                dtype=np.float32,
            )
            .reshape((1, 1))
            .astype(int)[0][0]  # type: ignore
        )
        detection_scores = np.frombuffer(
            response.raw_output_contents[output_index["detection_scores"]],
            dtype=np.float32,
        ).reshape((1, -1))[0]
        detection_classes = (
            np.frombuffer(
                response.raw_output_contents[output_index["detection_classes"]],
                dtype=np.float32,
            )
            .reshape((1, -1))
            .astype(int)  # type: ignore
        )[0]
        detection_boxes = np.frombuffer(
            response.raw_output_contents[output_index["detection_boxes"]],
            dtype=np.float32,
        ).reshape((1, -1, 4))[0]

        result = ObjectDetectionRawResult(
            num_detections=num_detections,
            detection_classes=detection_classes,
            detection_boxes=detection_boxes,
            detection_scores=detection_scores,
            detection_masks=None,
            label_names=self.label_names,
        )

        return result


class ObjectDetectionModelRegistry:
    models: dict[str, RemoteModel] = {}
    _loaded = False

    @classmethod
    def get_available_models(cls) -> list[str]:
        cls.load_all()
        return list(cls.models.keys())

    @classmethod
    def load(cls, name: str) -> RemoteModel:
        label_names = LABELS[name]
        model = RemoteModel(name, label_names)
        cls.models[name] = model
        return model

    @classmethod
    def get(cls, name: str) -> RemoteModel:
        if name not in cls.models:
            cls.load(name)
        return cls.models[name]
