from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PIL
import tqdm

from datasets import DatasetDict


def display_image_bounding_boxes(
    image: PIL.Image,
    bboxes: list[tuple[int, int, int, int]],
    ner_tags: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
):
    """Display an image with bounding boxes.

    :param image: the image
    :param bboxes: the bounding boxes
    """
    fig, ax = plt.subplots()
    ax.imshow(image)

    for i, bbox in enumerate(bboxes):
        color = colors[ner_tags[i]] if ner_tags else None
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

    if output_path:
        fig.savefig(output_path)
        plt.close()
    else:
        plt.show()


base_ds = DatasetDict.load_from_disk("datasets/ingredient-detection-layout-dataset-v1")


# Useful for debugging (checking if the image with bounding boxes is correct)
for split_name in ("train", "test"):
    root_path = Path(
        f"datasets/ingredient-detection-layout-dataset-v1/_output_images/{split_name}"
    )
    root_path.mkdir(parents=True, exist_ok=True)

    for item in tqdm.tqdm(base_ds[split_name], desc="dataset items"):
        if item["offsets"]:
            barcode = item["meta"]["barcode"]
            image_id = item["meta"]["image_id"]
            output_path = root_path / f"{barcode}_{image_id}.png"
            display_image_bounding_boxes(
                item["image"],
                item["bboxes"],
                item["ner_tags"],
                output_path=output_path,
                colors=["r", "g", "b"],
            )
