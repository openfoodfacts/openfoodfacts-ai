from collections import defaultdict
from pathlib import Path

import imagehash
import tqdm
from label_studio_sdk.client import LabelStudio
from openfoodfacts.utils import get_image_from_url, get_logger
from PIL import Image

logger = get_logger(__name__)


def check_ls_dataset(ls: LabelStudio, project_id: int):
    skipped = 0
    not_annotated = 0
    annotated = 0
    hash_map = defaultdict(list)
    for task in tqdm.tqdm(
        ls.tasks.list(project=project_id, fields="all"), desc="tasks"
    ):
        annotations = task.annotations

        if len(annotations) == 0:
            not_annotated += 1
            continue
        elif len(annotations) > 1:
            logger.warning("Task has multiple annotations: %s", task.id)
            continue

        annotation = annotations[0]

        if annotation["was_cancelled"]:
            skipped += 1

        annotated += 1
        image_url = task.data["image_url"]
        image = get_image_from_url(image_url)
        image_hash = str(imagehash.phash(image))
        hash_map[image_hash].append(task.id)

    for image_hash, task_ids in hash_map.items():
        if len(task_ids) > 1:
            logger.warning("Duplicate images: %s", task_ids)

    logger.info(
        "Tasks - annotated: %d, skipped: %d, not annotated: %d",
        annotated,
        skipped,
        not_annotated,
    )


def check_local_dataset(dataset_dir: Path, remove: bool = False):
    hash_map = defaultdict(list)
    for path in tqdm.tqdm(dataset_dir.glob("**/*.jpg"), desc="images"):
        if path.is_file() and path.suffix in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tiff",
            ".gif",
        ]:
            image = Image.open(path)
            image_hash = str(imagehash.phash(image))
            logger.debug("Image hash: %s", image_hash)
            hash_map[image_hash].append(path)

    duplicated = 0
    to_remove = []
    for image_hash, image_paths in hash_map.items():
        if len(image_paths) > 1:
            logger.warning(
                "Duplicate images: %s",
                [str(x.relative_to(dataset_dir)) for x in image_paths],
            )
            duplicated += 1
            to_remove.append(image_paths[0])

    logger.info("Total duplicated groups: %d", duplicated)

    if remove and to_remove:
        for path in to_remove:
            logger.info("Removing: %s", str(path))
            path.unlink()
