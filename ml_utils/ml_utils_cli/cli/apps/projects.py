import enum
import json
import typing
from pathlib import Path
from typing import Annotated, Optional

import typer
from openfoodfacts.utils import get_logger
from PIL import Image

from ..annotate import (
    format_annotation_results_from_triton,
    format_annotation_results_from_ultralytics,
)
from ..config import LABEL_STUDIO_DEFAULT_URL

app = typer.Typer()

logger = get_logger(__name__)


@app.command()
def create(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    title: Annotated[str, typer.Option(help="Project title")],
    config_file: Annotated[
        Path, typer.Option(help="Path to label config file", file_okay=True)
    ],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Create a new Label Studio project."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    label_config = config_file.read_text()

    project = ls.projects.create(title=title, label_config=label_config)
    logger.info(f"Project created: {project}")


@app.command()
def import_data(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    dataset_path: Annotated[
        Path, typer.Option(help="Path to the Label Studio dataset file", file_okay=True)
    ],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
    batch_size: int = 25,
):
    """Import tasks from a dataset file to a Label Studio project.

    The dataset file should contain one JSON object per line."""
    import more_itertools
    import tqdm
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    with dataset_path.open("rt") as f:
        for batch in more_itertools.chunked(
            tqdm.tqdm(map(json.loads, f), desc="tasks"), batch_size
        ):
            ls.projects.import_tasks(id=project_id, request=batch)


@app.command()
def update_prediction(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for task in ls.tasks.list(project=project_id, fields="all"):
        for prediction in task.predictions:
            prediction_id = prediction["id"]
            if prediction["model_version"] == "":
                logger.info("Updating prediction: %s", prediction_id)
                ls.predictions.update(
                    id=prediction_id,
                    model_version="undefined",
                )


@app.command()
def add_split(
    train_split: Annotated[
        float, typer.Option(help="fraction of samples to add in train split")
    ],
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Update the split field of tasks in a Label Studio project.

    The split field is set to "train" with probability `train_split`, and "val"
    otherwise. Tasks without a split field are assigned a split based on the
    probability, and updated in the server. Tasks with a non-null split field
    are not updated.
    """
    import random

    from label_studio_sdk import Task
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for task in ls.tasks.list(project=project_id, fields="all"):
        task: Task
        split = task.data.get("split")
        if split is None:
            split = "train" if random.random() < train_split else "val"
            logger.info("Updating task: %s, split: %s", task.id, split)
            ls.tasks.update(task.id, data={**task.data, "split": split})


@app.command()
def annotate_from_prediction(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    updated_by: Annotated[
        Optional[int], typer.Option(help="User ID to declare as annotator")
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Create annotations for all tasks from predictions.

    This command is useful if you imported tasks with predictions, and want to
    "validate" these predictions by creating annotations.
    """
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        task_id = task.id
        if task.total_annotations == 0 and task.total_predictions > 0:
            logger.info("Creating annotation for task: %s", task_id)
            ls.annotations.create(
                id=task_id,
                result=task.predictions[0]["result"],
                project=project_id,
                updated_by=updated_by,
            )


class PredictorBackend(enum.Enum):
    triton = "triton"
    ultralytics = "ultralytics"


@app.command()
def add_prediction(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of the object detection model to run (for Triton server) or "
            "of the Ultralytics zero-shot model to run."
        ),
    ] = "yolov8x-worldv2.pt",
    triton_uri: Annotated[
        Optional[str],
        typer.Option(help="URI (host+port) of the Triton Inference Server"),
    ] = None,
    backend: Annotated[
        PredictorBackend,
        typer.Option(
            help="Prediction backend: either use a Triton server to perform "
            "the prediction or uses Ultralytics."
        ),
    ] = PredictorBackend.ultralytics,
    labels: Annotated[
        Optional[list[str]],
        typer.Option(
            help="List of class labels to use for Yolo model. If you're using Yolo-World or other "
            "zero-shot models, this is the list of label names that are going to be provided to the "
            "model. In such case, you can use `label_mapping` to map the model's output to the "
            "actual class names expected by Label Studio."
        ),
    ] = None,
    label_mapping: Annotated[
        Optional[str],
        typer.Option(help="Mapping of model labels to class names, as a JSON string"),
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
    threshold: Annotated[
        Optional[float],
        typer.Option(
            help="Confidence threshold for selecting bounding boxes. The default is 0.5 "
            "for Triton backend and 0.1 for Ultralytics backend."
        ),
    ] = None,
    max_det: Annotated[int, typer.Option(help="Maximum numbers of detections")] = 300,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Launch in dry run mode, without uploading annotations to Label Studio"
        ),
    ] = False,
    error_raise: Annotated[
            bool,
            typer.Option(
                help="Raise an error if image download fails"
            ),
        ] = True,
    model_version: Annotated[
        Optional[str],
        typer.Option(help="Model version to use for the prediction"),
    ] = None,
):
    """Add predictions as pre-annotations to Label Studio tasks,
    for an object detection model running on Triton Inference Server."""

    import tqdm
    from cli.triton.object_detection import ObjectDetectionModelRegistry
    from label_studio_sdk.client import LabelStudio
    from openfoodfacts.utils import get_image_from_url

    label_mapping_dict = None
    if label_mapping:
        label_mapping_dict = json.loads(label_mapping)

    if dry_run:
        logger.info("** Dry run mode enabled **")

    logger.info(
        "backend: %s, model_name: %s, labels: %s, threshold: %s, label mapping: %s",
        backend,
        model_name,
        labels,
        threshold,
        label_mapping,
    )
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    model: ObjectDetectionModelRegistry | "YOLO"

    if backend == PredictorBackend.ultralytics:
        from ultralytics import YOLO

        if labels is None:
            raise typer.BadParameter("Labels are required for Ultralytics backend")

        if threshold is None:
            threshold = 0.1

        model = YOLO(model_name)
        if hasattr(model, "set_classes"):
            model.set_classes(labels)
        else:
            logger.warning("The model does not support setting classes directly.")
    elif backend == PredictorBackend.triton:
        if triton_uri is None:
            raise typer.BadParameter("Triton URI is required for Triton backend")

        if threshold is None:
            threshold = 0.5

        model = ObjectDetectionModelRegistry.load(model_name)
    else:
        raise typer.BadParameter(f"Unsupported backend: {backend}")

    for task in tqdm.tqdm(ls.tasks.list(project=project_id), desc="tasks"):
        if task.total_predictions == 0:
            image_url = task.data["image_url"]
            image = typing.cast(
                Image.Image,
                get_image_from_url(image_url, error_raise=error_raise),
            )
            if backend == PredictorBackend.ultralytics:
                results = model.predict(
                    image,
                    conf=threshold,
                    max_det=max_det,
                )[0]
                labels = typing.cast(list[str], labels)
                label_studio_result = format_annotation_results_from_ultralytics(
                    results, labels, label_mapping_dict
                )
            else:
                output = model.detect_from_image(image, triton_uri=triton_uri)
                results = output.select(threshold=threshold)
                logger.info("Adding prediction to task: %s", task.id)
                label_studio_result = format_annotation_results_from_triton(
                    results, image.width, image.height
                )
            if dry_run:
                logger.info("image_url: %s", image_url)
                logger.info("result: %s", label_studio_result)
            else:
                ls.predictions.create(
                    task=task.id,
                    result=label_studio_result,
                    model_version=model_version,
                )


@app.command()
def create_dataset_file(
    input_file: Annotated[
        Path,
        typer.Option(help="Path to a list of image URLs", exists=True),
    ],
    output_file: Annotated[
        Path, typer.Option(help="Path to the output JSON file", exists=False)
    ],
):
    """Create a Label Studio object detection dataset file from a list of
    image URLs."""
    from urllib.parse import urlparse

    import tqdm
    from cli.sample import format_object_detection_sample_to_ls
    from openfoodfacts.images import extract_barcode_from_url, extract_source_from_url
    from openfoodfacts.utils import get_image_from_url

    logger.info("Loading dataset: %s", input_file)

    with output_file.open("wt") as f:
        for line in tqdm.tqdm(input_file.open("rt"), desc="images"):
            url = line.strip()
            if not url:
                continue

            extra_meta = {}
            image_id = Path(urlparse(url).path).stem
            if ".openfoodfacts.org" in url:
                barcode = extract_barcode_from_url(url)
                extra_meta["barcode"] = barcode
                off_image_id = Path(extract_source_from_url(url)).stem
                extra_meta["off_image_id"] = off_image_id
                image_id = f"{barcode}-{off_image_id}"

            image = get_image_from_url(url, error_raise=False)

            if image is None:
                logger.warning("Failed to load image: %s", url)
                continue

            label_studio_sample = format_object_detection_sample_to_ls(
                image_id, url, image.width, image.height, extra_meta
            )
            f.write(json.dumps(label_studio_sample) + "\n")
