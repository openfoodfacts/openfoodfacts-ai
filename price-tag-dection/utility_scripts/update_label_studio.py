def fix_label(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        for prediction in task.predictions:
            updated = False
            if "result" in prediction:
                for result in prediction["result"]:
                    value = result["value"]
                    if "rectanglelabels" in value and value["rectanglelabels"] != [
                        "price-tag"
                    ]:
                        value["rectanglelabels"] = ["price-tag"]
                        updated = True

            if updated:
                print(f"Updating prediction {prediction['id']}, task {task.id}")
                ls.predictions.update(prediction["id"], result=prediction["result"])

        for annotation in task.annotations:
            updated = False
            if "result" in annotation:
                for result in annotation["result"]:
                    value = result["value"]
                    if "rectanglelabels" in value and value["rectanglelabels"] != [
                        "price-tag"
                    ]:
                        value["rectanglelabels"] = ["price-tag"]
                        updated = True

            if updated:
                print(f"Updating annotation {annotation['id']}, task {task.id}")
                ls.annotations.update(annotation["id"], result=annotation["result"])



def select_price_tag_images(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    project_id: Annotated[int, typer.Option(help="Label Studio project ID")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    import typing
    from pathlib import Path
    from typing import Any
    from urllib.parse import urlparse

    import requests
    import tqdm
    from label_studio_sdk.client import LabelStudio
    from label_studio_sdk.types.task import Task

    session = requests.Session()
    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    proof_paths = (Path(__file__).parent / "proof.txt").read_text().splitlines()
    task: Task
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, include="data,id"), desc="tasks"
    ):
        data = typing.cast(dict[str, Any], task.data)

        if "is_raw_product_shelf" in data:
            continue
        image_url = data["image_url"]
        file_path = urlparse(image_url).path.replace("/img/", "")
        r = session.get(
            f"https://robotoff.openfoodfacts.org/api/v1/images/predict?image_url={image_url}&models=price_proof_classification",
        )

        if r.status_code != 200:
            print(
                f"Failed to get prediction for {image_url}, error: {r.text} (status: {r.status_code})"
            )
            continue

        prediction = r.json()["predictions"]["price_proof_classification"][0]["label"]

        is_raw_preduct_shelf = False
        if prediction in ("PRICE_TAG", "SHELF"):
            is_raw_preduct_shelf = file_path in proof_paths

        ls.tasks.update(
            task.id,
            data={
                **data,
                "is_raw_product_shelf": "true" if is_raw_preduct_shelf else "false",
            },
        )