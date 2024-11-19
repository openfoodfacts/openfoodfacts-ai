import logging
from typing import Annotated

import tqdm
import typer
from label_studio_sdk.client import LabelStudio
from openfoodfacts.utils import get_logger

logger = get_logger(level=logging.WARNING)

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def run(
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
    project_id: int = 42,
    view_id: int = 61,
):
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=api_key)
    tasks = ls.tasks.list(project=project_id, fields="all", page_size=50, view=view_id)

    for task in tqdm.tqdm(tasks, desc="tasks"):
        if len(task.annotations) != 1:
            logger.error(f"Task {task.id} has {len(task.annotations)} annotations")
            continue

        annotation = task.annotations[0]
        issues_annotations = [
            r
            for r in annotation["result"]
            if r.get("type") == "choices" and r.get("from_name") == "issues"
        ]

        if len(issues_annotations) > 1:
            logger.error(f"Task {task.id} has more than one 'issues' annotations")
            continue
        if len(issues_annotations) == 0:
            continue

        issues = issues_annotations[0]["value"]["choices"]

        if "prepared-values" in issues:
            print(
                f"https://annotate.openfoodfacts.org/projects/42/data?tab=61&task={task.id}"
            )


if __name__ == "__main__":
    typer.run(run)
