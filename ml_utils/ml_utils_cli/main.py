from typing import Annotated, Optional

import typer
from cli.apps import datasets as dataset_app
from cli.apps import projects as project_app
from cli.apps import users as user_app
from cli.config import LABEL_STUDIO_DEFAULT_URL
from openfoodfacts.utils import get_logger

app = typer.Typer()

logger = get_logger()


@app.command()
def predict_object(
    model_name: Annotated[
        str, typer.Option(help="Name of the object detection model to run")
    ],
    image_url: Annotated[str, typer.Option(help="URL of the image to process")],
    triton_uri: Annotated[
        str, typer.Option(help="URI (host+port) of the Triton Inference Server")
    ],
    threshold: float = 0.5,
):
    from cli.triton.object_detection import ObjectDetectionModelRegistry
    from openfoodfacts.utils import get_image_from_url

    model = ObjectDetectionModelRegistry.get(model_name)
    image = get_image_from_url(image_url)
    output = model.detect_from_image(image, triton_uri=triton_uri)
    results = output.select(threshold=threshold)

    for result in results:
        typer.echo(result)


# Temporary scripts


@app.command()
def skip_rotated_images(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    updated_by: Annotated[
        Optional[int], typer.Option(help="User ID to declare as annotator")
    ] = None,
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    import requests
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task
    from openfoodfacts.ocr import OCRResult

    session = requests.Session()
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        if any(annotation["was_cancelled"] for annotation in task.annotations):
            continue

        assert task.total_annotations == 1, (
            "Task has multiple annotations (%s)" % task.id
        )
        task_id = task.id

        annotation = task.annotations[0]
        annotation_id = annotation["id"]

        ocr_url = task.data["image_url"].replace(".jpg", ".json")
        ocr_result = OCRResult.from_url(ocr_url, session=session, error_raise=False)

        if ocr_result is None:
            logger.warning("No OCR result for task: %s", task_id)
            continue

        orientation_result = ocr_result.get_orientation()

        if orientation_result is None:
            # logger.info("No orientation for task: %s", task_id)
            continue

        orientation = orientation_result.orientation.name
        if orientation != "up":
            logger.info(
                "Skipping rotated image for task: %s (orientation: %s)",
                task_id,
                orientation,
            )
            ls.annotations.update(
                id=annotation_id,
                was_cancelled=True,
                updated_by=updated_by,
            )
        elif orientation == "up":
            logger.debug("Keeping annotation for task: %s", task_id)


app.add_typer(user_app.app, name="users", help="Manage Label Studio users")
app.add_typer(
    project_app.app,
    name="projects",
    help="Manage Label Studio projects (create, import data, etc.)",
)
app.add_typer(
    dataset_app.app,
    name="datasets",
    help="Manage datasets (convert, export, check, etc.)",
)

if __name__ == "__main__":
    app()
