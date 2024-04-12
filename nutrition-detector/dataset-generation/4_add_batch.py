import random
from typing import Annotated

import tqdm
import typer
from label_studio_sdk import Client
from openfoodfacts.utils import get_logger

logger = get_logger()

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def assign_batch_to_samples(
    project_id: int,
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
    start_batch_id: int = 1,
    batch_size: int = 100,
):
    """Assign a batch (an integer starting from 1) to samples in the label
    studio project.

    All samples are fetched, sorted randomly and a unique batch number is
    assigned to each sample.
    """
    ls = Client(url=LABEL_STUDIO_URL, api_key=api_key)
    ls.check_connection()

    project = ls.get_project(project_id)
    tasks = project.get_tasks()
    logger.info(f"Found {len(tasks)} tasks in the project")
    # get tasks without batch ID
    tasks = [task for task in tasks if task["data"].get("batch") == "null"]
    # Sort tasks randomly
    random.shuffle(tasks)
    num_tasks_to_assign = (len(tasks) // batch_size) * batch_size
    logger.info(f"Assigning {num_tasks_to_assign} tasks to batches")
    tasks = tasks[:num_tasks_to_assign]

    for i, task in enumerate(tqdm.tqdm(tasks, desc="tasks")):
        batch_id = (i // batch_size) + start_batch_id
        batch = f"batch-{batch_id}"
        logger.info(f"Assigning task {task['id']} to batch {batch}")
        project.update_task(task["id"], data={**task["data"], "batch": batch})


if __name__ == "__main__":
    typer.run(assign_batch_to_samples)
