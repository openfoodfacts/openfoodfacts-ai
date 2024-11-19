from pathlib import Path
from typing import Annotated

import typer
from label_studio_sdk.client import LabelStudio
from openfoodfacts.utils import get_logger

logger = get_logger()

LABEL_STUDIO_URL = "https://annotate.openfoodfacts.org"


def create_project(
    api_key: Annotated[str, typer.Argument(envvar="LABEL_STUDIO_API_KEY")],
    label_config_path: Path = typer.Argument(
        file_okay=True, dir_okay=False, exists=True
    ),
    title: str = typer.Option(help="Project title"),
):
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=api_key)
    label_config = Path(label_config_path).read_text()

    project = ls.projects.create(title=title, label_config=label_config)
    logger.info(f"Project created: {project}")


if __name__ == "__main__":
    typer.run(create_project)
