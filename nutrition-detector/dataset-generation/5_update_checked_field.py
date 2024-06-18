from typing import Annotated

import typer
from label_studio_sdk import Client
from openfoodfacts.utils import get_logger

logger = get_logger()

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def update_checked_field(
    project_id: int,
    view_id: int,
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
):
    ls = Client(url=LABEL_STUDIO_URL, api_key=api_key)
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
