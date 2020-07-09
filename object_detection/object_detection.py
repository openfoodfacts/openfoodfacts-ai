import argparse
import dataclasses
import json
import pathlib
import re
import tempfile
import traceback
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from joblib import Parallel, delayed
import numpy as np
import PIL
from PIL import Image
import requests
import tensorflow as tf
import tqdm

import label_map_util


CategoryIndex = Dict[int, Dict]
JSONType = Dict[str, Any]


@dataclasses.dataclass
class ObjectDetectionResult:
    bounding_box: Tuple
    score: float
    label: str


@dataclasses.dataclass
class ObjectDetectionRawResult:
    num_detections: int
    detection_boxes: np.array
    detection_scores: np.array
    detection_classes: np.array
    category_index: CategoryIndex
    detection_masks: Optional[np.array] = None
    boxed_image: Optional[PIL.Image.Image] = None

    def select(self, threshold: Optional[float] = None) -> List[ObjectDetectionResult]:
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
            label_str = self.category_index[label_int]["name"]
            result = ObjectDetectionResult(
                bounding_box=tuple(bounding_box.tolist()),
                score=float(score),
                label=label_str,
            )
            results.append(result)

        return results

    def to_json(self, threshold: Optional[float] = None) -> List[JSONType]:
        return [dataclasses.asdict(r) for r in self.select(threshold)]


def convert_image_to_array(image: PIL.Image.Image) -> np.array:
    if image.mode != "RGB":
        image = image.convert("RGB")

    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def convert_image_to_array_batch(images: List[PIL.Image.Image]) -> Optional[np.array]:
    assert len(images) > 0

    if len(set((x.size for x in images))) != 1:
        return None

    images = [
        image.convert("RGB") if image.mode != "RGB" else image for image in images
    ]
    (im_width, im_height) = images[0].size

    for image in images:
        if isinstance(image, PIL.TiffImagePlugin.TiffImageFile):
            if not hasattr(image, "use_load_libtiff"):
                image.use_load_libtiff = True

    image_arrays = [
        np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        for image in images
    ]
    return np.stack(image_arrays, axis=0)


class ObjectDetectionModel:
    def __init__(self, graph: tf.Graph, label_map):
        self.graph: tf.Graph = graph
        self.label_map = label_map

        self.categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=1000
        )
        self.category_index: CategoryIndex = (
            label_map_util.create_category_index(self.categories)
        )

    @classmethod
    def load(cls, graph_path: pathlib.Path, label_path: pathlib.Path):
        print("Loading model...")
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(str(graph_path), "rb") as f:
                serialized_graph = f.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        label_map = label_map_util.load_labelmap(str(label_path))

        print("Model loaded")
        return cls(graph=detection_graph, label_map=label_map)

    def _run_inference_batch(
        self, batch_iter: Iterable[List[Tuple[str, str, Image.Image]]]
    ) -> Iterable[Tuple[str, str, ObjectDetectionRawResult, Image.Image]]:
        with tf.Session(graph=self.graph) as sess:
            # Get handles to input and output tensors
            ops = self.graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                "num_detections",
                "detection_boxes",
                "detection_scores",
                "detection_classes",
                "detection_masks",
            ]:
                tensor_name = key + ":0"
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)

            image_tensor = self.graph.get_tensor_by_name("image_tensor:0")

            for batch in batch_iter:
                images = convert_image_to_array_batch([x[2] for x in batch])

                if images is None:
                    continue

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: images})

                for i, image in enumerate(images):
                    # all outputs are float32 numpy arrays, so convert types as
                    # appropriate
                    detection_masks = (
                        output_dict["detection_masks"][i]
                        if "detection_masks" in output_dict
                        else None
                    )
                    raw_result = ObjectDetectionRawResult(
                        num_detections=int(output_dict["num_detections"][i]),
                        detection_classes=output_dict["detection_classes"][i].astype(
                            np.uint8
                        ),
                        detection_boxes=output_dict["detection_boxes"][i],
                        detection_scores=output_dict["detection_scores"][i],
                        detection_masks=detection_masks,
                        category_index=self.category_index,
                    )
                    yield (batch[i][0], batch[i][1], raw_result, image)


def iter_image_dimensions(file_path: pathlib.Path):
    with file_path.open("r") as f:
        for line in f:
            yield tuple(json.loads(line))


def get_image_from_url(
    image_url: str,
    error_raise: bool = False,
    session: Optional[requests.Session] = None,
) -> Optional[Image.Image]:
    try:
        if session:
            r = session.get(image_url)
        else:
            r = requests.get(image_url)
    except requests.exceptions.RequestException as e:
        if error_raise:
            raise e
        else:
            traceback.print_exc()
            return None

    if error_raise:
        r.raise_for_status()

    if r.status_code != 200:
        return None

    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        f.write(r.content)
        f.flush()
        try:
            image = Image.open(f.name)
        except IOError:
            return None

    image = image.convert("RGB") if image.mode != "RGB" else image

    try:
        image.load()
    except OSError as e:
        print(e)
        return None

    return image


def get_images_batch(
    batch: List[Tuple[str, str]], invalid_path: Optional[pathlib.Path]
) -> List[Tuple[str, str, Image.Image]]:
    image_urls = [
        generate_image_url(barcode, image_id) for (barcode, image_id) in batch
    ]
    images = Parallel(n_jobs=10)(
        delayed(get_image_from_url)(image_url) for image_url in image_urls
    )

    results = []
    invalid = []
    for item, image in zip(batch, images):
        if image is not None and image.width >= 100 and image.height >= 100:
            results.append((*item, image))
        else:
            invalid.append({"barcode": item[0], "image_id": item[1]})

    if invalid and invalid_path:
        with invalid_path.open("a") as f:
            for i in invalid:
                f.write("{}\n".format(json.dumps(i)))

    return results


BARCODE_PATH_REGEX = re.compile(r"^(...)(...)(...)(.*)$")


def split_barcode(barcode: str) -> List[str]:
    if not barcode.isdigit():
        raise ValueError("unknown barcode format: {}".format(barcode))

    match = BARCODE_PATH_REGEX.fullmatch(barcode)

    if match:
        return [x for x in match.groups() if x]

    return [barcode]


def generate_image_url(barcode: str, image_name: str) -> str:
    splitted_barcode = split_barcode(barcode)
    path = "/{}/{}.jpg".format("/".join(splitted_barcode), image_name)
    return "https://static.openfoodfacts.org/images/products" + path


def iter_images_batch(
    file_path: pathlib.Path,
    batch_size: int,
    seen_set: Set[Tuple[str, str]],
    invalid_path: Optional[pathlib.Path],
) -> Iterable[List[Tuple[str, str, Image.Image]]]:
    current_dim = None
    batch: List[Tuple[str, str]] = []

    for (width, height, barcode, image_id) in iter_image_dimensions(file_path):
        key = (barcode, image_id)

        if key in seen_set:
            continue

        dim = (width, height)

        if (current_dim is not None and dim != current_dim) or len(batch) >= batch_size:
            if len(batch) < batch_size:
                print("New image dimension: {}".format(dim))

            images_batch = get_images_batch(batch, invalid_path)

            if images_batch:
                yield images_batch

            if key not in seen_set:
                batch = [(barcode, image_id)]
        else:
            if key not in seen_set:
                batch.append((barcode, image_id))

        current_dim = dim

    if batch:
        images_batch = get_images_batch(batch, invalid_path)

        if images_batch:
            yield images_batch


def get_seen_set(file_path: pathlib.Path) -> Set[Tuple[str, str]]:
    seen_set: Set[Tuple[str, str]] = set()

    if not file_path.is_file():
        return seen_set

    with file_path.open("r") as f:
        for line in f:
            item = json.loads(line)
            seen_set.add((item["barcode"], item["image_id"]))

    return seen_set


def run_model(
    file_path: pathlib.Path,
    output_path: pathlib.Path,
    model: ObjectDetectionModel,
    batch_size: int,
    invalid_path: Optional[pathlib.Path],
):
    seen_set = get_seen_set(output_path)

    if invalid_path:
        seen_set = seen_set.union(get_seen_set(invalid_path))

    print("{} items in seen set".format(len(seen_set)))
    batch_iter = iter_images_batch(file_path, batch_size, seen_set, invalid_path)

    with output_path.open("a") as f:
        for barcode, image_id, result, image in tqdm.tqdm(
            model._run_inference_batch(batch_iter)
        ):
            output = {
                "barcode": barcode,
                "image_id": image_id,
                "result": result.to_json(threshold=0.1),
            }
            f.write(json.dumps(output) + "\n")
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=pathlib.Path)
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--invalid-path", type=pathlib.Path,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    graph_path = input_dir / "frozen_inference_graph.pb"
    label_path = input_dir / "labels.pbtxt"
    data_path = args.data_path
    model = ObjectDetectionModel.load(graph_path, label_path)
    invalid_path = args.invalid_path

    run_model(
        data_path,
        input_dir / "output.jsonl",
        model,
        batch_size=args.batch_size,
        invalid_path=invalid_path,
    )
