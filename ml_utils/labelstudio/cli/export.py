import functools
import logging
import pickle
import tempfile
from pathlib import Path

import datasets
import tqdm
from label_studio_sdk.client import LabelStudio
from openfoodfacts.images import download_image

from cli.sample import HF_DS_FEATURES, format_object_detection_sample_to_hf

logger = logging.getLogger(__name__)


def _pickle_sample_generator(dir: Path):
    """Generator that yields samples from pickles in a directory."""
    for pkl in dir.glob("*.pkl"):
        with open(pkl, "rb") as f:
            yield pickle.load(f)


def export_to_hf(
    ls: LabelStudio,
    repo_id: str,
    category_names: list[str],
    project_id: int,
):
    logger.info("Project ID: %d, category names: %s", project_id, category_names)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        logger.info("Saving samples to temporary directory: %s", tmp_dir)
        for split in ["train", "val"]:
            logger.info("Processing split: %s", split)
            for i, task in tqdm.tqdm(
                enumerate(ls.tasks.list(project=project_id, fields="all")),
                desc="tasks",
            ):
                if task.data["split"] != split:
                    continue
                sample = format_object_detection_sample_to_hf(
                    task.data, task.annotations, category_names
                )
                if sample is not None:
                    # Save output as pickle
                    with open(tmp_dir / f"{split}_{i:05}.pkl", "wb") as f:
                        pickle.dump(sample, f)

            hf_ds = datasets.Dataset.from_generator(
                functools.partial(_pickle_sample_generator, tmp_dir),
                features=HF_DS_FEATURES,
            )
            hf_ds.push_to_hub(repo_id, split=split)


def export_from_ls_to_ultralytics(
    ls: LabelStudio,
    output_dir: Path,
    category_names: list[str],
    project_id: int,
):
    """Export annotations from a Label Studio project to the Ultralytics
    format.

    The Label Studio project should be an object detection project with a
    single rectanglelabels annotation result per task.
    """
    logger.info("Project ID: %d, category names: %s", project_id, category_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        split_labels_dir = output_dir / "labels" / split
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        split_images_dir = output_dir / "images" / split
        split_images_dir.mkdir(parents=True, exist_ok=True)

    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"),
        desc="tasks",
    ):
        split = task.data["split"]
        if len(task.annotations) > 1:
            logger.warning("More than one annotation found, skipping")
            continue
        elif len(task.annotations) == 0:
            logger.debug("No annotation found, skipping")
            continue

        annotation = task.annotations[0]
        image_id = task.data["image_id"]

        image_url = task.data["image_url"]
        download_output = download_image(image_url, return_bytes=True)
        if download_output is None:
            logger.error("Failed to download image: %s", image_url)
            continue

        _, image_bytes = download_output

        with (output_dir / "images" / split / f"{image_id}.jpg").open("wb") as f:
            f.write(image_bytes)

        with (output_dir / "labels" / split / f"{image_id}.txt").open("w") as f:
            for annotation_result in annotation["result"]:
                if annotation_result["type"] != "rectanglelabels":
                    raise ValueError(
                        "Invalid annotation type: %s" % annotation_result["type"]
                    )

                value = annotation_result["value"]
                x_min = value["x"] / 100
                y_min = value["y"] / 100
                width = value["width"] / 100
                height = value["height"] / 100
                category_name = value["rectanglelabels"][0]
                category_id = category_names.index(category_name)

                # Save the labels in the Ultralytics format:
                # - one label per line
                # - each line is a list of 5 elements:
                #   - category_id
                #   - x_center
                #   - y_center
                #   - width
                #   - height
                x_center = x_min + width / 2
                y_center = y_min + height / 2
                f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    with (output_dir / "data.yaml").open("w") as f:
        f.write("path: .\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test:\n")
        f.write("names:\n")
        for i, category_name in enumerate(category_names):
            f.write(f"  {i}: {category_name}\n")


def export_from_hf_to_ultralytics(
    repo_id: str, output_dir: Path, download_images: bool = True
):
    """Export annotations from a Hugging Face dataset project to the
    Ultralytics format.

    The Label Studio project should be an object detection project with a
    single rectanglelabels annotation result per task.
    """
    logger.info("Repo ID: %s", repo_id)
    ds = datasets.load_dataset(repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    category_id_to_name = {}

    for split in ["train", "val"]:
        split_labels_dir = output_dir / "labels" / split
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        split_images_dir = output_dir / "images" / split
        split_images_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm.tqdm(ds[split], desc="samples"):
            image_id = sample["image_id"]
            image_url = sample["meta"]["image_url"]

            if download_images:
                download_output = download_image(image_url, return_bytes=True)
                if download_output is None:
                    logger.error("Failed to download image: %s", image_url)
                    continue
                _, image_bytes = download_output
                with (output_dir / "images" / split / f"{image_id}.jpg").open(
                    "wb"
                ) as f:
                    f.write(image_bytes)
            else:
                image = sample["image"]
                image.save(output_dir / "images" / split / f"{image_id}.jpg")

            objects = sample["objects"]
            bboxes = objects["bbox"]
            category_ids = objects["category_id"]
            category_names = objects["category_name"]

            with (output_dir / "labels" / split / f"{image_id}.txt").open("w") as f:
                for bbox, category_id, category_name in zip(
                    bboxes, category_ids, category_names
                ):
                    if category_id not in category_id_to_name:
                        category_id_to_name[category_id] = category_name
                    y_min, x_min, y_max, x_max = bbox
                    y_min = min(max(y_min, 0.0), 1.0)
                    x_min = min(max(x_min, 0.0), 1.0)
                    y_max = min(max(y_max, 0.0), 1.0)
                    x_max = min(max(x_max, 0.0), 1.0)
                    width = x_max - x_min
                    height = y_max - y_min
                    # Save the labels in the Ultralytics format:
                    # - one label per line
                    # - each line is a list of 5 elements:
                    #   - category_id
                    #   - x_center
                    #   - y_center
                    #   - width
                    #   - height
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

    category_names = [
        x[1] for x in sorted(category_id_to_name.items(), key=lambda x: x[0])
    ]
    with (output_dir / "data.yaml").open("w") as f:
        f.write("path: .\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test:\n")
        f.write("names:\n")
        for i, category_name in enumerate(category_names):
            f.write(f"  {i}: {category_name}\n")
