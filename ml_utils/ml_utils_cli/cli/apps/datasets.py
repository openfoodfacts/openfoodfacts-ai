import os
import json
import random
import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer
from openfoodfacts.utils import get_logger

from ..config import LABEL_STUDIO_DEFAULT_URL
from ..types import ExportDestination, ExportSource, TaskType
from ..annotate import MODEL_NAME, LABELS


app = typer.Typer()

logger = get_logger(__name__)


@app.command()
def check(
    api_key: Annotated[
        Optional[str], typer.Option(envvar="LABEL_STUDIO_API_KEY")
    ] = None,
    project_id: Annotated[
        Optional[int], typer.Option(help="Label Studio Project ID")
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
    dataset_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the dataset directory", exists=True, file_okay=False
        ),
    ] = None,
    remove: Annotated[
        bool,
        typer.Option(
            help="Remove duplicate images from the dataset, only for local datasets"
        ),
    ] = False,
):
    """Check a dataset for duplicate images."""
    from label_studio_sdk.client import LabelStudio

    from ..check import check_local_dataset, check_ls_dataset

    if project_id is not None:
        ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
        check_ls_dataset(ls, project_id)
    elif dataset_dir is not None:
        check_local_dataset(dataset_dir, remove=remove)
    else:
        raise typer.BadParameter("Either project ID or dataset directory is required")


@app.command()
def split_train_test(
    task_type: TaskType, dataset_dir: Path, output_dir: Path, train_ratio: float = 0.8
):
    """Split a dataset into training and test sets.

    Only classification tasks are supported.
    """
    if task_type == TaskType.classification:
        class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        logger.info("Found classes: %s", [d.name for d in class_dirs])

        output_dir.mkdir(parents=True, exist_ok=True)
        train_dir = output_dir / "train"
        test_dir = output_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for class_dir in class_dirs:
            input_paths = list(class_dir.glob("*"))
            random.shuffle(input_paths)

            test_count = int(len(input_paths) * (1 - train_ratio))
            if test_count == 0:
                logger.warning("Not enough samples, skipping class: %s", class_dir.name)
                continue

            test_paths = input_paths[:test_count]
            train_paths = input_paths[test_count:]

            for output_dir, input_paths in (
                (train_dir, train_paths),
                (test_dir, test_paths),
            ):
                output_cls_dir = output_dir / class_dir.name
                output_cls_dir.mkdir(parents=True, exist_ok=True)

                for path in input_paths:
                    logger.info("Copying: %s to %s", path, output_cls_dir)
                    shutil.copy(path, output_cls_dir / path.name)
    else:
        raise typer.BadParameter("Unsupported task type")


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
    from cli.sample import format_object_detection_sample_from_hf
    from datasets import load_dataset

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
    from_: Annotated[ExportSource, typer.Option("--from", help="Input source to use")],
    to: Annotated[ExportDestination, typer.Option(help="Where to export the data")],
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
    from cli.export import (
        export_from_hf_to_ultralytics,
        export_from_ls_to_ultralytics,
        export_to_hf,
    )
    from label_studio_sdk.client import LabelStudio

    if (to == ExportDestination.hf or from_ == ExportSource.hf) and repo_id is None:
        raise typer.BadParameter("Repository ID is required for export/import with HF")

    if to == ExportDestination.hf and category_names is None:
        raise typer.BadParameter("Category names are required for HF export")

    if from_ == ExportSource.ls:
        if project_id is None:
            raise typer.BadParameter("Project ID is required for LS export")
        if api_key is None:
            raise typer.BadParameter("API key is required for LS export")

    if to == ExportDestination.ultralytics and output_dir is None:
        raise typer.BadParameter("Output directory is required for Ultralytics export")

    if from_ == ExportSource.ls:
        ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
        category_names_list = category_names.split(",")
        if to == ExportDestination.hf:
            export_to_hf(ls, repo_id, category_names_list, project_id)
        elif to == ExportDestination.ultralytics:
            export_from_ls_to_ultralytics(
                ls, output_dir, category_names_list, project_id
            )

    elif from_ == ExportSource.hf:
        if to == ExportDestination.ultralytics:
            export_from_hf_to_ultralytics(
                repo_id, output_dir, download_images=download_images
            )
        else:
            raise typer.BadParameter("Unsupported export format")


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


@app.command()
def create_dataset_file_from_yolo(
    images_dir: Annotated[Path, typer.Option(exists=True)],
    output_file: Annotated[Path, typer.Option(exists=False)],
    model_name: str = MODEL_NAME,
    models_dir: str = "models", 
    labels: list[str] = LABELS,
    batch_size: int = 20,
):
    """Create a Label Studio object detection dataset file from a list of
    images."""
    from cli.annotate import format_object_detection_sample_from_yolo
    model_name = os.path.join(models_dir, model_name)
    samples = format_object_detection_sample_from_yolo(
        images_dir=images_dir, 
        model_name=model_name,
        labels=labels,
        batch_size=batch_size,
    )
    logger.info("Saving samples to %s", output_file)
    with output_file.open("wt") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")