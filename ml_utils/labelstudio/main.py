import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from openfoodfacts.images import extract_barcode_from_url, extract_source_from_url
from openfoodfacts.utils import get_logger

app = typer.Typer()

logger = get_logger()


LABEL_STUDIO_DEFAULT_URL = "https://annotate.openfoodfacts.org"


@app.command()
def create_project(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    title: Annotated[str, typer.Option(help="Project title")],
    config_file: Annotated[
        Path, typer.Option(help="Path to label config file", file_okay=True)
    ],
    label_studio_url: str = "https://annotate.openfoodfacts.org",
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
def convert_object_detection_dataset(
    repo_id: Annotated[
        str, typer.Option(help="Hugging Face Datasets repository ID to convert")
    ],
    output_file: Annotated[
        Path, typer.Option(help="Path to the output JSON file", exists=False)
    ],
):
    """Convert object detection dataset from Hugging Face Datasets to Label
    Studio format, and save it to a JSON file."""
    from datasets import load_dataset

    from cli.sample import format_object_detection_sample_from_hf

    logger.info("Loading dataset: %s", repo_id)
    ds = load_dataset(repo_id)
    logger.info("Dataset loaded: %s", tuple(ds.keys()))

    with output_file.open("wt") as f:
        for split in ds.keys():
            logger.info("Processing split: %s", split)
            for sample in ds[split]:
                label_studio_sample = format_object_detection_sample_from_hf(
                    sample, split=split
                )
                f.write(json.dumps(label_studio_sample) + "\n")


@app.command()
def export(
    from_: Annotated[str, typer.Option("--from", help="Input format to use")],
    to: Annotated[str, typer.Option(help="Export format to use")],
    api_key: Annotated[Optional[str], typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    repo_id: Annotated[
        Optional[str],
        typer.Option(help="Hugging Face Datasets repository ID to convert"),
    ] = None,
    category_names: Annotated[
        Optional[str],
        typer.Option(help="Category names to use, as a comma-separated list"),
    ] = None,
    project_id: Annotated[
        Optional[int], typer.Option(help="Label Studio Project ID")
    ] = None,
    label_studio_url: Optional[str] = LABEL_STUDIO_DEFAULT_URL,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Path to the output directory", file_okay=False),
    ] = None,
    download_images: Annotated[
        bool,
        typer.Option(
            help="if True, don't use HF images and download images from the server"
        ),
    ] = False,
):
    """Export Label Studio annotation, either to Hugging Face Datasets or
    local files (ultralytics format)."""
    from label_studio_sdk.client import LabelStudio

    from cli.export import (
        export_from_hf_to_ultralytics,
        export_from_ls_to_ultralytics,
        export_to_hf,
    )

    if (to == "hf" or from_ == "hf") and repo_id is None:
        raise typer.BadParameter("Repository ID is required for export/import with HF")

    if to == "hf" and category_names is None:
        raise typer.BadParameter("Category names are required for HF export")

    if from_ == "ls":
        if project_id is None:
            raise typer.BadParameter("Project ID is required for LS export")
        if api_key is None:
            raise typer.BadParameter("API key is required for LS export")

    if to == "ultralytics" and output_dir is None:
        raise typer.BadParameter("Output directory is required for Ultralytics export")

    if from_ == "ls":
        if to == "hf":
            ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
            category_names_list = category_names.split(",")
            export_to_hf(ls, repo_id, category_names_list, project_id)
        elif to == "ultralytics":
            export_from_ls_to_ultralytics(
                ls, output_dir, category_names_list, project_id
            )
        else:
            raise typer.BadParameter("Unsupported export format")

    elif from_ == "hf":
        if to == "ultralytics":
            export_from_hf_to_ultralytics(
                repo_id, output_dir, download_images=download_images
            )
        else:
            raise typer.BadParameter("Unsupported export format")
    else:
        raise typer.BadParameter("Unsupported input format")


@app.command()
def check_dataset(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Check a dataset for duplicate images."""
    from collections import defaultdict

    import imagehash
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from openfoodfacts.utils import get_image_from_url

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    skipped = 0
    not_annotated = 0
    annotated = 0
    hash_map = defaultdict(list)
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        annotations = task.annotations

        if len(annotations) == 0:
            not_annotated += 1
            continue
        elif len(annotations) > 1:
            logger.warning("Task has multiple annotations: %s", task.id)
            continue

        annotation = annotations[0]

        if annotation["was_cancelled"]:
            skipped += 1

        annotated += 1
        # print(task)
        image_url = task.data["image_url"]
        image = get_image_from_url(image_url)
        image_hash = str(imagehash.phash(image))
        hash_map[image_hash].append(task.id)

    for image_hash, task_ids in hash_map.items():
        if len(task_ids) > 1:
            logger.warning("Duplicate images: %s", task_ids)

    logger.info(
        "Tasks - annotated: %d, skipped: %d, not annotated: %d",
        annotated,
        skipped,
        not_annotated,
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
    """Create a Label Studio dataset file from a list of image URLs."""
    from urllib.parse import urlparse

    import tqdm
    from openfoodfacts.utils import get_image_from_url

    from cli.sample import format_object_detection_sample_to_ls

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


@app.command()
def predict_object(
    model_name: Annotated[
        str, typer.Option(help="Name of the object detection model to run")
    ],
    image_url: Annotated[str, typer.Option(help="URL of the image to process")],
    triton_uri: Annotated[
        str, typer.Option(help="URI (host+port) of the Triton Inference Server")
    ],
    threshold: float = 0.5,
):
    from openfoodfacts.utils import get_image_from_url

    from cli.triton.object_detection import ObjectDetectionModelRegistry

    model = ObjectDetectionModelRegistry.get(model_name)
    image = get_image_from_url(image_url)
    output = model.detect_from_image(image, triton_uri=triton_uri)
    results = output.select(threshold=threshold)

    for result in results:
        typer.echo(result)


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
    from label_studio_sdk.client import LabelStudio
    from openfoodfacts.utils import get_image_from_url

    from cli.sample import format_annotation_results_from_triton
    from cli.triton.object_detection import ObjectDetectionModelRegistry

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

    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for task in ls.tasks.list(project=project_id, fields="all"):
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


# Label Studio user management


@app.command()
def list_user(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """List all users in Label Studio."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for user in ls.users.list():
        print(f"{user.id:02d}: {user.email}")


@app.command()
def delete_user(
    user_id: int,
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Delete a user from Label Studio."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    ls.users.delete(user_id)


# Temporary scripts


@app.command()
def skip_rotated_images(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    updated_by: Annotated[
        Optional[int], typer.Option(help="User ID to declare as annotator")
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    import requests
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task
    from openfoodfacts.ocr import OCRResult

    session = requests.Session()
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        if any(annotation["was_cancelled"] for annotation in task.annotations):
            continue

        assert task.total_annotations == 1, (
            "Task has multiple annotations (%s)" % task.id
        )
        task_id = task.id

        annotation = task.annotations[0]
        annotation_id = annotation["id"]

        ocr_url = task.data["image_url"].replace(".jpg", ".json")
        ocr_result = OCRResult.from_url(ocr_url, session=session, error_raise=False)

        if ocr_result is None:
            logger.warning("No OCR result for task: %s", task_id)
            continue

        orientation_result = ocr_result.get_orientation()

        if orientation_result is None:
            # logger.info("No orientation for task: %s", task_id)
            continue

        orientation = orientation_result.orientation.name
        if orientation != "up":
            logger.info(
                "Skipping rotated image for task: %s (orientation: %s)",
                task_id,
                orientation,
            )
            ls.annotations.update(
                id=annotation_id,
                was_cancelled=True,
                updated_by=updated_by,
            )
        elif orientation == "up":
            logger.debug("Keeping annotation for task: %s", task_id)


if __name__ == "__main__":
    app()
