"""Push dataset to Hugging Face dataset hub."""

import functools
import pickle
import tempfile
from pathlib import Path
from typing import Annotated, Iterator, Optional
from urllib.parse import urlparse

import datasets
import tqdm
import typer
from label_studio_sdk import Task
from label_studio_sdk.client import LabelStudio
from openfoodfacts.images import extract_barcode_from_url
from openfoodfacts.utils import get_image_from_url, get_logger

logger = get_logger()

app = typer.Typer(pretty_exceptions_enable=False)


def create_sample(task: Task, only_checked: bool = False) -> Optional[dict]:
    """Generate a LayoutLM dataset sample from a Label Studio task.

    :param item: the Label Studio task
    :param only_checked: whether to include only tasks that have been checked
        by a second annotator.
    :return: the LayoutLMv3 dataset sample, or None if the task is not
        valid
    """
    task_data = task.data
    annotations_data = task.annotations
    task_id = task.id
    if len(annotations_data) == 0:
        # No annotation, skip
        return None
    elif len(annotations_data) > 1:
        logger.error("Task %s has more than one annotation", task_id)
        return None

    annotation_data = annotations_data[0]
    annotation_results = annotation_data["result"]

    ocr_url = task_data["meta"]["ocr_url"]
    image_url = task_data["image_url"]
    meta = {
        "barcode": extract_barcode_from_url(image_url),
        "image_id": Path(urlparse(image_url).path).stem,
        "ocr_url": ocr_url,
        "image_url": task_data["image_url"],
        "batch": task_data["batch"],
        "label_studio_id": task_id,
        "checked": False,
        "usda_table": False,
        "nutrition_text": False,
        "no_nutrition_table": False,
        "comment": "",
    }

    if task_data.get("error", "ok") != "ok":
        logger.warning("Task %s has an error: %s", task_id, task_data["error"])
        return None

    current_bbox_id = None
    tokens = []
    bboxes = []
    ner_tags = []
    for result in annotation_results:
        result_value = result["value"]
        if result["from_name"] in ("transcription", "label"):
            if current_bbox_id is None:
                current_bbox_id = result["id"]
            elif current_bbox_id != result["id"]:
                logger.error(
                    "Mismatch in bbox ids: %s != %s", current_bbox_id, result["id"]
                )
                return None
            # There are only two types of results of interest: labels and
            # textarea
            current_bbox_id = None

            if result["from_name"] == "label":
                assert len(result_value["labels"]) == 1, "Only one label is expected"
                ner_tag = result_value["labels"][0]
                if ner_tag in ("other", "other-nutriment"):
                    ner_tag = "O"
                else:
                    ner_tag = ner_tag.replace("-", "_").upper()
                    previous_ner_tag = ner_tags[-1] if ner_tags else "O"
                    previous_ner_tag = (
                        previous_ner_tag
                        if previous_ner_tag == "O"
                        else previous_ner_tag.split("-")[1]
                    )
                    if previous_ner_tag == ner_tag:
                        ner_tag = f"I-{ner_tag}"
                    else:
                        ner_tag = f"B-{ner_tag}"
                ner_tags.append(ner_tag)
            elif result["from_name"] == "transcription":
                assert len(result_value["text"]) == 1, "Only one text is expected"
                tokens.append(result_value["text"][0])
                # Also add bounding box information
                # Every coordinate is between 0 and 100 (excluded), LayoutLM
                # requires an integer between 0 and 1000 (excluded) for the
                # dataset
                x_min = int(result_value["x"] * 10)
                y_min = int(result_value["y"] * 10)
                x_max = int((result_value["x"] + result_value["width"]) * 10)
                y_max = int((result_value["y"] + result_value["height"]) * 10)
                bboxes.append(
                    (
                        max(0, min(999, x_min)),
                        max(0, min(999, y_min)),
                        max(0, min(999, x_max)),
                        max(0, min(999, y_max)),
                    )
                )

        elif result["type"] == "choices":
            if result["from_name"] == "info":
                info_checkbox = result["value"]["choices"]
                for info_name in (
                    "checked",
                    "usda-table",
                    "nutrition-text",
                    "no-nutrition-table",
                ):
                    meta[info_name.replace("-", "_")] = info_name in info_checkbox
            elif result["from_name"] == "issues" and result["value"]["choices"]:
                logger.info(
                    "Task %s has issues: %s, skipping",
                    task_id,
                    result["value"]["choices"],
                )
                return None
        elif result["type"] == "rectangle":
            # ignore the rectangle results, as they don't provide any useful
            # information
            continue
        elif result["from_name"] == "comment":
            meta["comment"] = result["value"]["text"][0]
        else:
            logger.warning("Unknown result type: %s", result["type"])
            continue

    if only_checked and not meta["checked"]:
        logger.info("Task %s is not checked, skipping", task_id)
        return None

    if not (len(tokens) == len(bboxes) == len(ner_tags)):
        logger.warning(
            "Mismatch in lengths for task ID %s: [tokens]:%s [bboxes]:%s [ner_tags]:%s",
            task_id,
            len(tokens),
            len(bboxes),
            len(ner_tags),
        )
        return None

    image = get_image_from_url(image_url, error_raise=False)

    if image is None:
        logger.info("Cannot load image from %s, skipping", image_url)
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")

    return {
        "meta": meta,
        "tokens": tokens,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
        "image": image,
    }


def get_tasks(
    label_studio_url: str, api_key: str, project_id: int, batch_ids: list[int] = None
) -> Iterator[dict]:
    """Yield tasks (annotations) from Label Studio."""

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    filter_items = [
        {
            "filter": "filter:tasks:completed_at",
            "operator": "empty",
            "type": "Datetime",
            "value": False,
        }
    ]

    if batch_ids is not None:
        filter_items.append(
            {
                "filter": "filter:tasks:data.batch",
                "operator": "regex",
                "type": "Unknown",
                "value": "batch-{}$".format("|".join(map(str, batch_ids))),
            }
        )
    yield from ls.tasks.list(
        project=project_id,
        query={
            "filters": {
                "conjunction": "and",
                "items": filter_items,
            }
        },
        # This view contains all samples
        view=61,
        fields="all",
    )


def sample_generator(dir_path: Path):
    for file_path in dir_path.iterdir():
        with file_path.open("rb") as f:
            yield pickle.load(f)


@app.command()
def push_dataset(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    # The project ID is hardcoded to 42, as it is the ID of the project on our
    # Label Studio instance
    project_id: Annotated[int, typer.Option(..., help="Label Studio project ID")] = 42,
    batch_ids: Annotated[
        Optional[str],
        typer.Option(..., help="comma-separated list of batch IDs to include"),
    ] = None,
    label_studio_url: Annotated[
        str, typer.Option()
    ] = "https://annotate.openfoodfacts.org",
    revision: Annotated[
        str, typer.Option(help="Dataset revision on Hugging Face datasets")
    ] = "main",
    test_split_count: Annotated[
        int, typer.Option(help="Number of samples in test split")
    ] = 180,
):
    logger.info("Fetching tasks from Label Studio, project %s", project_id)
    if batch_ids:
        batch_ids = list(map(int, batch_ids.split(",")))
        logger.info("Fetching tasks for batches %s", batch_ids)

    created = 0
    ner_tag_set = set()

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        logger.info("Saving samples in temporary directory %s", tmp_dir)

        for i, task in enumerate(
            tqdm.tqdm(
                get_tasks(label_studio_url, api_key, project_id, batch_ids),
                desc="tasks",
            )
        ):
            sample = create_sample(task)
            if sample:
                logger.debug("Sample: %s", sample)
                created += 1
                ner_tag_set.update(
                    ner_tag.split("-", maxsplit=1)[1]
                    for ner_tag in sample["ner_tags"]
                    if ner_tag != "O"
                )
                with (tmp_dir / f"{i:04d}.pkl").open("wb") as f:
                    pickle.dump(sample, f)

    logger.info("Generated %s samples", created)

    if not created:
        logger.error("No valid samples found, exiting")
        raise typer.Exit(code=1)

    all_ner_tags = ["O"]
    for ner_tag in ner_tag_set:
        all_ner_tags.extend([f"B-{ner_tag}", f"I-{ner_tag}"])

    logger.info("NER tags: %s", all_ner_tags)
    features = datasets.Features(
        {
            "ner_tags": datasets.Sequence(
                datasets.features.ClassLabel(names=all_ner_tags)
            ),
            "tokens": datasets.Sequence(datasets.Value("string")),
            "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
            "image": datasets.features.Image(),
            "meta": {
                "barcode": datasets.Value("string"),
                "image_id": datasets.Value("string"),
                "image_url": datasets.Value("string"),
                "ocr_url": datasets.Value("string"),
                "batch": datasets.Value("string"),
                "label_studio_id": datasets.Value("int64"),
                "checked": datasets.Value("bool"),
                "usda_table": datasets.Value("bool"),
                "nutrition_text": datasets.Value("bool"),
                "no_nutrition_table": datasets.Value("bool"),
                "comment": datasets.Value("string"),
            },
        }
    )
    dataset = datasets.Dataset.from_generator(
        functools.partial(sample_generator, tmp_dir), features=features
    )
    dataset = dataset.train_test_split(test_size=test_split_count, shuffle=False)

    # dataset.save_to_disk("datasets/nutrient-detection-layout")
    logger.info(
        "Pushing dataset to Hugging Face Hub under openfoodfacts/nutrient-detection-layout, revision %s",
        revision,
    )
    dataset.push_to_hub("openfoodfacts/nutrient-detection-layout", revision=revision)


if __name__ == "__main__":
    app()
