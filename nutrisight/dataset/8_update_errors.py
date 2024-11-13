from typing import Annotated

import tqdm
import typer
from label_studio_sdk import Client
from openfoodfacts.utils import get_logger

logger = get_logger(level="DEBUG")

LABEL_STUDIO_DEFAULT_URL = "https://annotate.openfoodfacts.org"


def update_checked_field(
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
    project_id: int = 42,
    view_id: int = 62,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    ls = Client(url=label_studio_url, api_key=api_key)
    ls.check_connection()

    project = ls.get_project(project_id)
    tasks = project.get_tasks(view_id=view_id)
    logger.info(f"Found {len(tasks)} tasks with errors in the project")
    for task in tqdm.tqdm(tasks, desc="tasks"):
        data = task["data"]
        project.update_task(task["id"], data={**data})


if __name__ == "__main__":
    typer.run(update_checked_field)
