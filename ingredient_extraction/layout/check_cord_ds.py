from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import PIL
from matplotlib import patches
from transformers import AutoModelForTokenClassification, AutoProcessor

from datasets import load_dataset


def display_image_bounding_boxes_cord(
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
        color = colors[ner_tags[i]] if ner_tags else "r"
        x_min, y_min, x_max, y_max = bbox
        x_min *= image.width / 1000
        y_min *= image.height / 1000
        x_max *= image.width / 1000
        y_max *= image.height / 1000
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


processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=7
)
dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")


example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
word_labels = example["ner_tags"]

display_image_bounding_boxes_cord(image, boxes)
