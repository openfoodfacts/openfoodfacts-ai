import json
import random
import shutil
import typing
from pathlib import Path
from typing import Annotated, Optional

import typer
from openfoodfacts.utils import get_logger

from ..config import LABEL_STUDIO_DEFAULT_URL
from ..types import ExportDestination, ExportSource, TaskType

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
    label_names: Annotated[
        Optional[str],
        typer.Option(help="Label names to use, as a comma-separated list"),
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
    train_ratio: Annotated[
        float,
        typer.Option(
            help="Train ratio for splitting the dataset, if the split name is not "
            "provided (typically, if the source is Label Studio)"
        ),
    ] = 0.8,
    error_raise: Annotated[
        bool,
        typer.Option(
            help="Raise an error if an image download fails, only for Ultralytics"
        ),
    ] = True,
):
    """Export Label Studio annotation, either to Hugging Face Datasets or
    local files (ultralytics format)."""
    from cli.export import (
        export_from_hf_to_ultralytics,
        export_from_ls_to_hf,
        export_from_ls_to_ultralytics,
    )
    from label_studio_sdk.client import LabelStudio

    if (to == ExportDestination.hf or from_ == ExportSource.hf) and repo_id is None:
        raise typer.BadParameter("Repository ID is required for export/import with HF")

    if label_names is None:
        if to == ExportDestination.hf:
            raise typer.BadParameter("Label names are required for HF export")
        if from_ == ExportSource.ls:
            raise typer.BadParameter(
                "Label names are required for export from LS source"
            )

    if from_ == ExportSource.ls:
        if project_id is None:
            raise typer.BadParameter("Project ID is required for LS export")
        if api_key is None:
            raise typer.BadParameter("API key is required for LS export")

    if to == ExportDestination.ultralytics and output_dir is None:
        raise typer.BadParameter("Output directory is required for Ultralytics export")

    if from_ == ExportSource.ls:
        ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
        label_names = typing.cast(str, label_names)
        label_names_list = label_names.split(",")
        if to == ExportDestination.hf:
            repo_id = typing.cast(str, repo_id)
            export_from_ls_to_hf(
                ls, repo_id, label_names_list, typing.cast(int, project_id)
            )
        elif to == ExportDestination.ultralytics:
            export_from_ls_to_ultralytics(
                ls,
                typing.cast(Path, output_dir),
                label_names_list,
                typing.cast(int, project_id),
                train_ratio=train_ratio,
                error_raise=error_raise,
            )

    elif from_ == ExportSource.hf:
        if to == ExportDestination.ultralytics:
            export_from_hf_to_ultralytics(
                typing.cast(str, repo_id),
                typing.cast(Path, output_dir),
                download_images=download_images,
                error_raise=error_raise,
            )
        else:
            raise typer.BadParameter("Unsupported export format")
