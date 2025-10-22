# /// script
# dependencies = [
#   "duckdb",
#   "typer",
#   "openfoodfacts",
#   "tqdm",
#   "Pillow",
#   "ultralytics",
#   "albumentations",
# ]
# ///
import random
import typing
from pathlib import Path

import duckdb
import tqdm
import typer
from openfoodfacts.images import ImageDownloadItem, download_image

app = typer.Typer()


def _download_front_images(parquet_path: Path, output_dir: Path, count: int = 1000):
    result = duckdb.sql(
        """SELECT code, images
        FROM read_parquet($parquet_path)
        WHERE len(images) > 0 AND list_contains(states_tags, 'en:front-photo-selected')
        ORDER BY RANDOM()
        LIMIT ($limit)
        """,
        params={"parquet_path": str(parquet_path), "limit": count},
    )
    downloaded = 0
    result_iter = tqdm.tqdm(result.fetchall(), desc="products")
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in result_iter:
        code, images = row
        front_images = [image for image in images if image["key"].startswith("front")]

        if front_images:
            random.shuffle(front_images)
            front_image = front_images[0]
            image_id = front_image["imgid"]
            image_key = front_image["key"]
            # typer.echo(f"Downloading {image_key} for code {code}")
            image_struct = typing.cast(
                ImageDownloadItem, download_image((code, image_id), return_struct=True)
            )

            if image_struct.image_bytes:
                downloaded += 1
                result_iter.set_description(f"products ({downloaded} downloaded)")

                batch_dirname = f"batch-{downloaded // 100}"
                batch_dir = output_dir / batch_dirname
                batch_dir.mkdir(parents=True, exist_ok=True)
                output_path = batch_dir / f"{code}_{image_id}_{image_key}.jpg"
                # typer.echo(f"Saving to {output_path}")
                with open(output_path, "wb") as f:
                    f.write(image_struct.image_bytes)


def _download_other_images(parquet_path: Path, output_dir: Path, count: int = 1000):
    result = duckdb.sql(
        """SELECT code, images
        FROM read_parquet($parquet_path)
        WHERE len(images) > 0
        ORDER BY RANDOM()
        LIMIT ($limit)
        """,
        params={"parquet_path": str(parquet_path), "limit": count},
    )
    downloaded = 0
    result_iter = tqdm.tqdm(result.fetchall(), desc="products")
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in result_iter:
        code, images = row
        # image IDs of front images
        front_image_id = set(
            int(image["imgid"]) for image in images if image["key"].startswith("front")
        )

        # only keep "raw" images (digit keys) that are not selected as front
        # images
        other_images = [
            image
            for image in images
            if image["key"].isdigit() and int(image["key"]) not in front_image_id
        ]

        if other_images:
            random.shuffle(other_images)
            front_image = other_images[0]
            image_id = front_image["key"]
            # typer.echo(f"Downloading {image_key} for code {code}")
            image_struct = typing.cast(
                ImageDownloadItem, download_image((code, image_id), return_struct=True)
            )

            if image_struct.image_bytes:
                downloaded += 1
                result_iter.set_description(f"products ({downloaded} downloaded)")

                batch_dirname = f"batch-{downloaded // 100}"
                batch_dir = output_dir / batch_dirname
                batch_dir.mkdir(parents=True, exist_ok=True)
                output_path = batch_dir / f"{code}_{image_id}.jpg"
                # typer.echo(f"Saving to {output_path}")
                with open(output_path, "wb") as f:
                    f.write(image_struct.image_bytes)


@app.command()
def download_images(
    parquet_path: Path,
    output_dir: Path,
    front_image_count: int = 1000,
    other_image_count: int = 1000,
):
    """Create a dataset by downloading images from Open Food Facts and
    pre-annotating them using Open Food Facts data.

    We create a dataset to train an image classifier model that can
    distinguish between front images and other images of products.

    We use images selected as "front" images in Open Food Facts as
    positive examples, and other images that were not selected as
    front images as negative examples.

    We create two directories in the output directory:
    - `front`: contains front images of products
    - `other`: contains other images of products

    Images are grouped in batches of 100 images each, and saved in
    subdirectories named `batch-0`, `batch-1`, etc. in the respective
    directories. This makes it easier to review the dataset in
    """
    front_image_dir = output_dir / "front"
    front_image_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading front images to {front_image_dir}")
    _download_front_images(
        parquet_path=parquet_path,
        output_dir=output_dir,
        count=front_image_count,
    )
    _download_other_images(
        parquet_path=parquet_path,
        output_dir=output_dir / "other",
        count=other_image_count,
    )


@app.command()
def create_dataset(
    input_dir: Path,
    output_dir: Path,
    front_train_batch_ids: str = typer.Option(
        help="Batch IDs to include in the train dataset for front images (comma-separated)",
    ),
    other_train_batch_ids: str = typer.Option(
        help="Batch IDs to include in the train dataset for other images (comma-separated)",
    ),
    front_test_batch_ids: str = typer.Option(
        help="Batch IDs to include in the test dataset for front images (comma-separated)",
    ),
    other_test_batch_ids: str = typer.Option(
        help="Batch IDs to include in the test dataset for other images (comma-separated)",
    ),
):
    """Create a dataset by combining front and other images from the
    specified batches.

    The input directory should contain two subdirectories:
    - `front`: contains front images of products
    - `other`: contains other images of products

    The output directory will contain the combined dataset.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, class_name, batch_ids in [
        ("train", "front", front_train_batch_ids),
        ("train", "other", other_train_batch_ids),
        ("test", "front", front_test_batch_ids),
        ("test", "other", other_test_batch_ids),
    ]:
        if not batch_ids:
            typer.echo(f"No batch IDs provided for {split} {class_name} images")
            continue

        class_output_dir = output_dir / split / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(
            f"Creating dataset for {split} {class_name} images in {class_output_dir}"
        )
        batch_ids_list = [
            batch_id.strip() for batch_id in batch_ids.split(",") if batch_id.strip()
        ]
        for batch_id in batch_ids_list:
            batch_dir = input_dir / class_name / f"batch-{batch_id}"
            if batch_dir.exists():
                for image_path in batch_dir.glob("*.jpg"):
                    output_path = class_output_dir / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    # Create a symlink to the image in the output directory
                    typer.echo(f"Linking {image_path} to {output_path}")
                    output_path.symlink_to(image_path)
            else:
                typer.echo(f"Batch {batch_id} not found in {class_name} images")


@app.command()
def train(
    dataset_dir: Path,
    model_name: str = "yolo11n-cls.pt",
    epochs: int = 10,
    imgsz: int = 224,
    batch: int = 64,
):
    """Train the image classifier model using Ultralytics."""
    import ultralytics

    model = ultralytics.YOLO(model_name)
    model.train(data=dataset_dir, epochs=epochs, imgsz=imgsz, batch=batch)


@app.command()
def predict(
    dataset_dir: Path,
    output_dir: Path,
    model_path: Path,
    group_by_confidence: bool = typer.Option(
        False,
        help="Group predictions by confidence level (0.1 buckets).",
    ),
    add_prob_suffix: bool = typer.Option(
        False,
        help="Add the top1 confidence probability to the output filename.",
    ),
    copy_images: bool = typer.Option(
        False, help="Copy images instead of creating symlinks."
    ),
):
    """Predict the class of a directory of images using a trained model.

    We use a custom classification predictor so that we can apply custom
    pre-processing.

    Args:
        dataset_dir: Directory containing images to classify.
        output_dir: Directory to save the classification results.
        model_path: Path to the trained model.
    """
    import ultralytics

    from ml_commons import CustomClassificationPredictor

    model = ultralytics.YOLO(model_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    for result in model.predict(
        source=dataset_dir, predictor=CustomClassificationPredictor
    ):
        image_path = Path(result.path)
        top1conf = result.probs.top1conf.item()
        predicted_class = result.names[result.probs.top1]
        bucket_id = (top1conf // 0.1) / 10
        output_path = output_dir / predicted_class

        if group_by_confidence:
            output_path = output_path / str(bucket_id)

        suffix = f"_{top1conf}" if add_prob_suffix else ""
        output_path = output_path / f"{image_path.stem}{suffix}{image_path.suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if copy_images:
            typer.echo(f"Copying {image_path} to {output_path}")
            output_path.write_bytes(image_path.read_bytes())
        else:
            output_path.symlink_to(image_path)


if __name__ == "__main__":
    app()
