from typing import Annotated

import typer
from label_studio_sdk import Task
from label_studio_sdk.client import LabelStudio
from openfoodfacts.utils import get_logger

logger = get_logger(level="DEBUG")

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def add_checked_field(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    view_id: Annotated[int, typer.Option(help="Label Studio view ID")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in ls.tasks.list(project=project_id, fields="all", view=view_id):
        if task.annotations and "checked" not in task.data:
            last_annotation_results = task.annotations[-1]["result"]
            for annotation_result in last_annotation_results:
                if (
                    annotation_result["type"] == "choices"
                    and "checked" in annotation_result["value"]["choices"]
                ):
                    logger.info(f"Updating task {task['id']} with checked field")
                    ls.tasks.update(task.id, data={**task.data, "checked": True})
                    break


if __name__ == "__main__":
    typer.run(add_checked_field)
