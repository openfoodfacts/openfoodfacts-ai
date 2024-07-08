import random
from typing import Annotated

import tqdm
import typer
from label_studio_sdk import Client
from label_studio_sdk.client import LabelStudio
from openfoodfacts.utils import get_logger

logger = get_logger()

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def assign_batch_to_samples(
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
    project_id: int = 42,
    start_batch_id: int = 1,
    batch_size: int = 100,
):
    """Assign a batch (an integer starting from 1) to samples in the label
    studio project.

    All samples are fetched, sorted randomly and a unique batch number is
    assigned to each sample.
    """
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=api_key)
    tasks = list(ls.tasks.list(project=project_id, include="data,id", page_size=500))
    logger.info(f"Found {len(tasks)} tasks in the project")
    # get tasks without batch ID
    tasks = [task for task in tasks if task.data.get("batch") == "null"]
    logger.info(f"Found {len(tasks)} tasks without batch ID")
    breakpoint()
    # Sort tasks randomly
    random.shuffle(tasks)
    num_tasks_to_assign = (len(tasks) // batch_size) * batch_size
    logger.info(f"Assigning {num_tasks_to_assign} tasks to batches")
    tasks = tasks[:num_tasks_to_assign]

    for i, task in enumerate(tqdm.tqdm(tasks, desc="tasks")):
        batch_id = (i // batch_size) + start_batch_id
        batch = f"batch-{batch_id}"
        logger.info(f"Assigning task {task.id} to batch {batch}")
        ls.tasks.update(id=task.id, data={**task.data, "batch": batch})


if __name__ == "__main__":
    typer.run(assign_batch_to_samples)
