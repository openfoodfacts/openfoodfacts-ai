"""Check for errors in the dataset."""

import copy
import math
from typing import Annotated, Optional

import typer
from label_studio_sdk import Client
from openfoodfacts.utils import get_logger

logger = get_logger()


def is_bounding_box_modified(word_annotation: dict, word_prediction: dict):
    if word_annotation.keys() != word_prediction.keys():
        logger.info(
            "Keys are different: %s / %s",
            word_annotation.keys(),
            word_prediction.keys(),
        )
        return True

    for key in word_annotation.keys():
        if key == "value":
            for subkey in word_annotation[key].keys():
                if isinstance(
                    word_annotation[key][subkey], (float, int)
                ) and isinstance(word_prediction[key][subkey], (float, int)):
                    if not math.isclose(
                        word_annotation[key][subkey], word_prediction[key][subkey]
                    ):
                        return True
                elif word_annotation[key][subkey] != word_prediction[key][subkey]:
                    return True
        else:
            if word_annotation[key] != word_prediction[key]:
                return True

    return False


def check_bounding_boxes_unmodified(annotation: dict, prediction: dict) -> str | None:
    prediction_results = prediction["result"]
    annotation_results = copy.deepcopy(
        [
            result
            for result in annotation["result"]
            # only keep the bounding boxes + transcription + labels
            if result["from_name"] not in ("info", "issues", "comment")
        ]
    )

    for result in annotation_results:
        # Remove "origin": "prediction" key-value pair
        result.pop("origin", None)
        # Remove "labels" key-value pair so that we can compare the results
        if result["from_name"] == "label":
            result["value"]["labels"] = None

    for result in prediction_results:
        if result["from_name"] == "label":
            result["value"]["labels"] = None

    if len(prediction_results) != len(annotation_results):
        error_message = f"number of prediction bounding boxes is different from number of annotation bounding boxes  ({len(prediction_results)} / {len(annotation_results)})"
        print(f"Error: {error_message}")
        return error_message

    diff_count = sum(
        int(is_bounding_box_modified(a, p))
        for p, a in zip(prediction_results, annotation_results)
    )
    if diff_count:
        error_message = f"some prediction bounding boxes are different from annotation bounding boxes (diff: {diff_count} / {len(annotation_results)})"
        print(f"Error: {error_message}")

        if diff_count <= 5:
            for p, a in zip(prediction_results, annotation_results):
                if is_bounding_box_modified(a, p):
                    print(f"Prediction: {p}")
                    print(f"Annotation: {a}")
                    print("---" * 10)
        return error_message

    return None


def check_task(task: dict) -> Optional[dict]:
    logger.info(
        "Checking task https://annotate.openfoodfacts.org/projects/42/data?tab=55&task=%s",
        task["id"],
    )
    annotations_data = task["annotations"]
    prediction_data = task["predictions"][0]

    if len(annotations_data) == 0:
        # No annotation, skip
        return None
    elif len(annotations_data) > 1:
        logger.error("Task %s has more than one annotation", task["id"])
        return None

    annotation_data = annotations_data[0]
    check_bounding_boxes_unmodified(annotation_data, prediction_data)


def get_tasks(
    label_studio_url: str, api_key: str, project_id: int, batch_ids: list[int] = None
):
    """Get tasks (annotations) from Label Studio."""
    ls = Client(url=label_studio_url, api_key=api_key)
    ls.check_connection()
    project = ls.get_project(project_id)

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
    return project.get_tasks(
        filters={
            "conjunction": "and",
            "items": filter_items,
        },
        # This view contains all samples
        view_id=61,
    )


def check_errors(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    # The project ID is hardcoded to 42, as it is the ID of the project on our
    # Label Studio instance
    project_id: Annotated[int, typer.Option(..., help="Label Studio project ID")] = 42,
    batch_ids: Optional[list[int]] = None,
    label_studio_url: Annotated[
        str, typer.Option()
    ] = "https://annotate.openfoodfacts.org",
):
    logger.info("Fetching tasks from Label Studio, project %s", project_id)
    tasks = get_tasks(label_studio_url, api_key, project_id, batch_ids)

    for task in tasks:
        check_task(task)


if __name__ == "__main__":
    # typer.run(check_errors)
    import os

    check_errors(os.environ["LABEL_STUDIO_API_KEY"])
