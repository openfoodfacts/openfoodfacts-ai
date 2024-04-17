import copy
import os
from collections import Counter

import functions_framework
from label_studio_sdk import Client, Project

LABEL_STUDIO_URL = os.environ["LABEL_STUDIO_URL"]
LABEL_STUDIO_API_KEY = os.environ["LABEL_STUDIO_API_KEY"]
PROJECT_ID = os.environ["LABEL_STUDIO_PROJECT_ID"]


@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function to process Label Studio webhooks.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    if request.method != "POST":
        return {
            "status": "error",
            "message": "Only POST requests are supported",
        }

    request_json = request.get_json(silent=True)
    if not request_json:
        return {
            "status": "error",
            "message": "Request payload is empty",
        }

    if "action" not in request_json:
        return {
            "status": "error",
            "message": "Action not found in the request payload",
        }

    action = request_json["action"]
    if action not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED", "ANNOTATION_DELETED"):
        return {
            "status": "error",
            "message": f"Action {action} is not supported",
        }

    ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    project = ls.get_project(PROJECT_ID)
    annotation = request_json["annotation"]

    if action == "ANNOTATION_DELETED":
        return process_annotation_deleted(project, annotation)

    return process_annotation_created_or_updated(project, annotation)


def process_annotation_deleted(project: Project, annotation: dict) -> dict:
    task_id = annotation["task"]
    task = project.get_tasks(selected_ids=[task_id])[0]
    task_data = task["data"]
    task_data["checked"] = False
    task_data["error"] = "ok"
    task_data["warning"] = "ok"
    print(f"Reseting checked flag from task {task_id}")
    project.update_task(task_id, data=task_data)
    return {"status": "deleted"}


def process_annotation_created_or_updated(project: Project, annotation: dict) -> dict:
    task_id = annotation["task"]
    task = project.get_tasks(selected_ids=[task_id])[0]
    task_data = task["data"]
    task_data["checked"] = compute_checked_flag(annotation)
    task_data["error"] = check_for_errors(annotation)
    task_data["warning"] = check_for_warnings(annotation)
    project.update_task(task_id, data=task_data)
    return {"status": "updated"}


def compute_checked_flag(annotation: dict) -> bool:
    """Add the `checked` flag to the task data if the annotation has the `checked` value added by
    the annotator."""
    checked = False
    for annotation_result in annotation["result"]:
        if (
            annotation_result["type"] == "choices"
            and "checked" in annotation_result["value"]["choices"]
        ):
            checked = True
            break
    return checked


def check_for_errors(annotation: dict) -> str:
    """ "Add the `error` field to the task data.

    Two types of errors are checked:
    - whether the bounding boxes are modified (moved, resized, etc.)
    - whether some bounding boxes are deleted
    """
    prediction = annotation.get("prediction")
    if not prediction:
        print("Error: no prediction found in the annotation")
        return "no prediction found in the annotation"

    for result in annotation["result"]:
        if result["from_name"] in "issues":
            # If there is an issue, we don't perform error check
            return "ok"

    if error := check_bounding_boxes_unmodified(annotation, prediction):
        return error

    return "ok"


def check_for_warnings(annotation: dict) -> str:
    """Add the `warning` field to the task data.

    The following warnings are checked:
    - whether the same label is annotated more than 3 times
    - whether only a few labels of the same class (_100g or _serving) are
      annotated
    - whether the serving-size label is annotated when serving labels are
      annotated
    """
    for result in annotation["result"]:
        if result["from_name"] in "issues":
            # If there is an issue, we don't perform warning check
            return "ok"

    if warning := check_warning_annotated_values(annotation):
        return warning

    return "ok"


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
        int(p != a) for p, a in zip(prediction_results, annotation_results)
    )
    if diff_count:
        error_message = f"some prediction bounding boxes are different from annotation bounding boxes (diff: {diff_count} / {len(annotation_results)})"
        print(f"Error: {error_message}")

        if diff_count <= 5:
            for p, a in zip(prediction_results, annotation_results):
                if p != a:
                    print(f"Prediction: {p}")
                    print(f"Annotation: {a}")
                    print("---" * 10)
        return error_message

    return None


def check_warning_annotated_values(annotation: dict) -> str | None:
    annotation_results = [
        result
        for result in annotation["result"]
        # only keep the label
        if result["from_name"] == "label"
    ]
    labels = [result["value"]["labels"][0] for result in annotation_results]
    label_counts = Counter(labels)

    serving_distinct_nutriment_count = 0
    per_100g_distinct_nutriment_count = 0
    for k, count in label_counts.items():
        if count > 3 and k not in ("other", "other-nutriment"):
            warning_message = (
                f"label {k} is annotated more than 3 times ({count} times)"
            )
            print(f"Warning: {warning_message}")
            return warning_message

        if k.endswith("_serving"):
            serving_distinct_nutriment_count += 1
        elif k.endswith("_100g"):
            per_100g_distinct_nutriment_count += 1

    if per_100g_distinct_nutriment_count > 0 and per_100g_distinct_nutriment_count < 7:
        warning_message = (
            f"only {per_100g_distinct_nutriment_count} per 100g labels are annotated"
        )
        print(f"Warning: {warning_message}")
        return warning_message

    if serving_distinct_nutriment_count > 0 and serving_distinct_nutriment_count < 7:
        warning_message = (
            f"only {serving_distinct_nutriment_count} per serving labels are annotated"
        )
        print(f"Warning: {warning_message}")
        return warning_message

    if serving_distinct_nutriment_count > 0 and "serving-size" not in label_counts:
        warning_message = (
            "serving-size is not annotated when serving labels are annotated"
        )
        print(f"Warning: {warning_message}")
        return warning_message

    return None
