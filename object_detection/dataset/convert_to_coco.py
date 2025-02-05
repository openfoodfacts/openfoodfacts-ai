""""This script shows how to convert from an object detection dataset in the
Hugging Face Datasets format to the COCO format."""

import json
from pathlib import Path

from datasets import Dataset, load_dataset


def convert_to_coco(hf_ds: Dataset, output_dir: Path) -> dict:
    coco_ds = {
        "info": {
            "year": "2024",
            "version": "1",
            "description": "",
            "contributor": "Open Food Facts",
            "url": "",
            "date_created": "2000-01-01T00:00:00+00:00",
        },
        "licenses": {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by-sa/3.0/deed.en",
            "name": "BY SA 3.0",
        },
    }

    categories = []
    images = []
    annotations = []

    for sample in hf_ds:
        image = sample["image"]
        assert image.format == "JPEG", f"Invalid format: {image.format}"
        assert image.mode == "RGB", f"Invalid mode: {image.mode}"
        image_width = sample["width"]
        image_height = sample["height"]

        with open(output_dir / f"{sample['image_id']}.jpg", "wb") as f:
            image.save(f, format="JPEG")

        images.append(
            {
                "id": sample["image_id"],
                "license": 1,
                "file_name": f"{sample['image_id']}.jpg",
                "height": image_height,
                "width": image_width,
            }
        )
        objects = sample["objects"]
        for i in range(len(objects["bbox"])):
            bbox = objects["bbox"][i]
            category_id = objects["category_id"][i]
            category_name = objects["category_name"][i]

            if category_name not in [x["name"] for x in categories]:
                categories.append(
                    {"id": category_id, "name": category_name, "supercategory": "none"}
                )
            # original data is in the format (ymin, xmin, ymax, xmax) with
            # relative coordinates
            # Convert to coco format (xmin, ymin, width, height) with absolute
            # coordinates
            coco_bbox = [
                bbox[1] * image_width,
                bbox[0] * image_height,
                (bbox[3] - bbox[1]) * image_width,
                (bbox[2] - bbox[0]) * image_height,
            ]
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": sample["image_id"],
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0,
                }
            )

    categories.sort(key=lambda x: x["id"])
    coco_ds["categories"] = categories
    coco_ds["images"] = images
    coco_ds["annotations"] = annotations

    with (output_dir / "annotations.json").open("w") as f:
        json.dump(coco_ds, f, indent=2)


hf_ds = load_dataset("openfoodfacts/nutriscore-object-detection")
output_dir = Path("nutriscore-object-detection")
output_dir.mkdir(exist_ok=True, parents=True)
for split in ["train", "test"]:
    split_dir = output_dir / split
    split_dir.mkdir(exist_ok=True, parents=True)
    convert_to_coco(hf_ds[split], split_dir)
