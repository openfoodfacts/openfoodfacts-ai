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
def create_export_snapshot(
    title: Annotated[str, typer.Option(help="Export snapshot title")],
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Create an export snapshot for a Label Studio project."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    output = ls.projects.exports.create(id=project_id, request={"title": title})
    logger.info(f"Export snapshot created: {output}")


@app.command()
def list_export_snapshot(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """List export snapshot for a Label Studio project."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    logger.info("## Exported snapshot ##")
    for snapshot in ls.projects.exports.list(id=project_id):
        logger.info(
            f"id: {snapshot.id}, title: {snapshot.title}, status: {snapshot.status}, converted_formats: {snapshot.converted_formats}"
        )


@app.command()
def download_export_snapshot(
    export_id: Annotated[int, typer.Option(help="Export snapshot ID")],
    export_type: Annotated[str, typer.Option(help="Export format to use")],
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """List export snapshot for a Label Studio project."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    snapshot = ls.projects.exports.download(
        id=project_id, export_pk=export_id, export_type=export_type
    )
    logger.info(f"Export snapshot downloaded: {snapshot}")


@app.command()
def convert_object_detection_dataset(
    dataset_id: Annotated[
        str, typer.Option(help="Hugging Face Datasets dataset ID to convert")
    ],
    output_file: Annotated[
        Path, typer.Option(help="Path to the output JSON file", exists=False)
    ],
):
    """Convert object detection dataset from Hugging Face Datasets to Label
    Studio format, and save it to a JSON file."""
    from datasets import load_dataset

    from cli.sample import format_object_detection_sample_from_hf

    logger.info("Loading dataset: %s", dataset_id)
    ds = load_dataset(dataset_id)

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
    to: Annotated[str, typer.Option(help="Export format to use")],
    repo_id: Annotated[
        str, typer.Option(help="Hugging Face Datasets repository ID to convert")
    ],
    category_names: Annotated[
        str, typer.Option(help="Category names to use, as a comma-separated list")
    ],
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Path to the output directory", file_okay=False),
    ] = None,
):
    from label_studio_sdk.client import LabelStudio

    from cli.export import export_to_hf, export_to_ultralytics

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    category_names_list = category_names.split(",")

    if to == "hf":
        export_to_hf(ls, repo_id, category_names_list, project_id)

    elif to == "ultralytics":
        if output_dir is None:
            raise typer.BadParameter(
                "Output directory is required for Ultralytics export"
            )

        export_to_ultralytics(ls, output_dir, category_names_list, project_id)


@app.command()
def check_dataset(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio Project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
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
    from urllib.parse import urlparse

    import tqdm
    from openfoodfacts.utils import get_image_from_url

    from cli.sample import format_object_detection_sample

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

            label_studio_sample = format_object_detection_sample(
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
    train_split: float,
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
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
def list_user(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
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


if __name__ == "__main__":
    app()
