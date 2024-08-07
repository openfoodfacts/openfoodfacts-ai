import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from openfoodfacts.utils import get_logger

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


@app.command()
def add_prediction(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    model_name: Annotated[
        str, typer.Option(help="Name of the object detection model to run")
    ],
    triton_uri: Annotated[
        str, typer.Option(help="URI (host+port) of the Triton Inference Server")
    ],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
    threshold: float = 0.5,
):
    """Add predictions as pre-annotations to Label Studio tasks,
    for an object detection model running on Triton Inference Server."""

    import tqdm
    from cli.sample import format_annotation_results_from_triton
    from cli.triton.object_detection import ObjectDetectionModelRegistry
    from label_studio_sdk.client import LabelStudio
    from openfoodfacts.utils import get_image_from_url

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    model = ObjectDetectionModelRegistry.load(model_name)

    for task in tqdm.tqdm(ls.tasks.list(project=project_id), desc="tasks"):
        if task.total_predictions == 0:
            image = get_image_from_url(task.data["image_url"], error_raise=True)
            output = model.detect_from_image(image, triton_uri=triton_uri)
            results = output.select(threshold=threshold)
            logger.info("Adding prediction to task: %s", task.id)
            label_studio_result = format_annotation_results_from_triton(
                results, image.width, image.height
            )
            ls.predictions.create(
                task=task.id,
                result=label_studio_result,
            )
