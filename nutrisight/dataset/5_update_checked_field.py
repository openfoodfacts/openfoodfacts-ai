from typing import Annotated

import typer
from label_studio_sdk import Client
from openfoodfacts.utils import get_logger

logger = get_logger()

LABEL_STUDIO_DEFAULT_URL = "https://annotate.openfoodfacts.org"


def update_checked_field(
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
    project_id: int = 42,
    view_id: int = 64,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
) -> None:
    """The `checked` field is a boolean field that indicates if the task has
    been checked by the annotator. When the second annotator marks the task
    as checked (this information is saved in the annotation result), a Google
    Cloud Function is triggered to update the task in the Label Studio project
    so that the `data.checked` field is set to True. This allows us to filter
    out the tasks that have not been checked yet in the Label Studio UI.

    This script is used to update the `data.checked` field when the Google
    Cloud Function failed for some reason to update the task in the Label
    Studio project.

    Args:
        api_key (str): The API key for the Label Studio project.
        project_id (int): The ID of the Label Studio project.
        view_id (int): The ID of the Label Studio view.
        label_studio_url (str): The URL of the Label Studio instance.
    """
    ls = Client(url=label_studio_url, api_key=api_key)
    ls.check_connection()

    project = ls.get_project(project_id)
    tasks = project.get_tasks(view_id=view_id)
    logger.info(f"Found {len(tasks)} tasks in the project")
    for task in tasks:
        data = task["data"]
        annotations = task["annotations"]
        if annotations and "checked" not in data:
            last_annotation_results = annotations[-1]["result"]
            for annotation_result in last_annotation_results:
                if (
                    annotation_result["type"] == "choices"
                    and "checked" in annotation_result["value"]["choices"]
                ):
                    logger.info(f"Updating task {task['id']} with checked field")
                    project.update_task(task["id"], data={**data, "checked": True})
                    break


if __name__ == "__main__":
    typer.run(update_checked_field)
