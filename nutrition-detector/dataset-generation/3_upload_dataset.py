import json
from pathlib import Path
from typing import Annotated

import tqdm
import typer
from label_studio_sdk.client import LabelStudio
from more_itertools import chunked

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def upload_dataset(
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
    project_id: int = 42,
):
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=api_key)

    with Path("./dataset.jsonl").open() as f:
        for batch in chunked(tqdm.tqdm(map(json.loads, f), desc="tasks"), 25):
            ls.projects.import_tasks(id=project_id, request=batch)


if __name__ == "__main__":
    typer.run(upload_dataset)
