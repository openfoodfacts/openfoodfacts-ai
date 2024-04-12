from pathlib import Path
from typing import Annotated

import typer
from label_studio_sdk import Client
from openfoodfacts.utils import get_logger

logger = get_logger()

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def create_project(
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")]
):
    ls = Client(url=LABEL_STUDIO_URL, api_key=api_key)
    ls.check_connection()

    label_config = Path("./label_config.xml").read_text()

    project = ls.start_project(
        title="Nutrition table token annotation",
        label_config=label_config,
    )
    logger.info(f"Project created: {project}")


if __name__ == "__main__":
    typer.run(create_project)
