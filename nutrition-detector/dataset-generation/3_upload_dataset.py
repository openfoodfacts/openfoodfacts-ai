import json
from pathlib import Path
from typing import Annotated

import tqdm
import typer
from label_studio_sdk import Client
from more_itertools import chunked

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def upload_dataset(
    project_id: int,
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
):
    ls = Client(url=LABEL_STUDIO_URL, api_key=api_key)
    ls.check_connection()

    project = ls.get_project(project_id)
    project.delete_all_tasks()

    with Path("./dataset.jsonl").open() as f:
        for batch in chunked(tqdm.tqdm(map(json.loads, f), desc="tasks"), 25):
            project.import_tasks(batch)


if __name__ == "__main__":
    typer.run(upload_dataset)
