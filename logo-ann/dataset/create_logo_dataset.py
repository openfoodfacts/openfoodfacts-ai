import argparse
import io
import itertools
import json
import operator
from pathlib import Path
from typing import Optional

import requests
import tqdm
from PIL import Image

ALLOWED_TYPES = {
    "label",
    "brand",
    "store",
    "packager_code",
    "qr_code",
    "packaging",
    "packager_code",
}


def is_valid_item(item):
    if item["annotation_type"] not in ALLOWED_TYPES:
        return False

    if item["annotation_type"] in ("label", "brand") and not item["taxonomy_value"]:
        return False

    if (
        item["annotation_type"] in ("store", "packaging")
        and not item["annotation_value_tag"]
    ):
        return False

    return True


def item_group_func(x):
    return (
        x["annotation_type"],
        (
            x["taxonomy_value"]
            if x["annotation_type"] in ("label", "brand")
            else x["annotation_value_tag"]
        )
        or "",
    )


def filter_items(items, min_count: int = 0):
    valid_items = [item for item in items if is_valid_item(item)]
    for key, group in itertools.groupby(
        sorted(valid_items, key=item_group_func), item_group_func,
    ):
        group_items = list(group)
        if len(group_items) >= min_count:
            print(key, len(group_items))
            yield from group_items


def get_image_from_url(
    image_url: str,
    error_raise: bool = False,
    session: Optional[requests.Session] = None,
) -> Optional[Image.Image]:
    if session:
        r = session.get(image_url)
    else:
        r = requests.get(image_url)

    if error_raise:
        r.raise_for_status()

    if r.status_code != 200:
        return None

    return Image.open(io.BytesIO(r.content))


def save_logo(item, image: Image.Image, output_dir: Path):
    annotation_type = item["annotation_type"]
    dir_name = annotation_type

    if annotation_type in ("label", "brand"):
        dir_name += f"_{item['taxonomy_value']}"
    elif annotation_type in ("store", "packaging"):
        dir_name += f"_{item['annotation_value_tag']}"

    dir_name = dir_name.replace(":", "_").replace("/", "-")
    dir_path = output_dir / dir_name
    dir_path.mkdir(exist_ok=True)

    source_image = item["source_image"]
    image_id = source_image.split("/")[-1].split(".")[0]
    file_path = dir_path / f"{item['barcode']}_{image_id}_{item['id']}.png"

    y_min, x_min, y_max, x_max = item["bounding_box"]
    (left, right, top, bottom) = (
        x_min * image.width,
        x_max * image.width,
        y_min * image.height,
        y_max * image.height,
    )
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.convert("RGB").save(file_path)


parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--use-cache", action="store_true")
args = parser.parse_args()
assert args.input.is_file()
args.output_dir.mkdir(parents=True, exist_ok=True)

session = requests.Session()

with args.input.open("r") as f:
    items = list(map(json.loads, f))

filtered_items = list(filter_items(items, min_count=5))
print(f"Items: {len(items)}")
print(f"Filtered items: {len(filtered_items)}")


seen_images = set()

CACHE_FILE = Path(__file__).parent / ".processed_images.txt"
if args.use_cache:
    if CACHE_FILE.is_file():
        with CACHE_FILE.open("r") as f:
            for line in f:
                seen_images.add(line.strip())


total_groups = len(set(x["source_image"] for x in filtered_items))

with CACHE_FILE.open("a") as f:
    for source_image, group in tqdm.tqdm(
        itertools.groupby(
            sorted(filtered_items, key=operator.itemgetter("source_image")),
            operator.itemgetter("source_image"),
        ), total=total_groups
    ):
        if source_image in seen_images:
            print(f"Skipping {source_image}")
            continue

        group = list(group)

        if not len(group):
            raise ValueError

        image_url = f"https://world.openfoodfacts.org/images/products{source_image}"

        try:
            image = get_image_from_url(image_url, error_raise=True, session=session)
        except requests.exceptions.HTTPError as e:
            if "404 Client Error: Not Found" in str(e):
                print(f"404 Error: {image_url}")
                continue

        for item in group:
            save_logo(item, image, args.output_dir)

        f.write(f"{source_image}\n")
