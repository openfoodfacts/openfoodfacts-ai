# /// script
# dependencies = [
#   "typer>=0.15.1",
#   "tqdm",
#   "Pillow",
#   "google-genai",
#   "google-cloud-storage",
#   "pydantic>=2.0",
#   "duckdb",
#   "openfoodfacts",
#   "numpy",
#   "pandas",
#   "orjson",
#   "gcloud-aio-storage",
#   "aiohttp",
#   "aiofiles",
# ]
# ///


import asyncio
import copy
import csv
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import aiofiles
import aiohttp
import orjson
import typer
from gcloud.aio.storage import Storage
from google.genai.types import JSONSchema as GoogleJSONSchema
from google.genai.types import Schema as GoogleSchema
from openfoodfacts.images import generate_image_url
from openfoodfacts.types import Flavor
from pydantic import BaseModel, Field, ValidationError
from tqdm.asyncio import tqdm

app = typer.Typer()


class CategoryInfo(BaseModel):
    category: str = Field(..., description="Name of the category of the product.")
    language: str = Field(
        ...,
        description="Language (ISO 639-1 code) associated with the category. Example: if the `category` is in English, it should be 'en'.",
    )


class CategoryPredictionResponseModel(BaseModel):
    categories: list[CategoryInfo] = Field(
        ...,
        description="List of categories associated with product. The most precise category should be returned first, then broader categories if possible. "
        "Prefer categories in English, unless there there are no good translations in English available. Don't return duplicate categories in different languages. "
        "If no category could not be determined, return an empty list.",
    )
    type: Literal[
        "food",
        "beauty",
        "petfood",
        "medication",
        "nutrition-supplement",
        "household",
        "product",
        "not-a-product",
    ] = Field(
        ...,
        description="Type of product being categorized. If the image does not contain a product, use 'not-a-product'. "
        "If the product does not belong to any of the other categories, use 'product'.",
    )
    explanation: str = Field(
        ...,
        description="Explanation of the reasoning behind the category prediction.",
    )


DEFAULT_INSTRUCTIONS = "Predict the categories of this product."


def get_image_urls(
    barcode: str, images: list[dict[str, Any]], flavor: Flavor
) -> list[str]:
    """Generate image URLs that will be used for product categorization.

    We look for the most recent front and ingredients images, and return them
    in that order. If only one of them is available, we return just that one.

    Args:
        barcode (str): The barcode of the product.
        images (list[dict[str, Any]]): List of image metadata dictionaries, as
            available in the Parquet export.
        flavor (Flavor): The flavor of the image to generate URLs for.

    """
    images = copy.deepcopy(images)
    for image in images:
        if not image["key"].isdigit():
            # Not a raw image, get the uploaded_t from the raw image
            raw_image_id = image["imgid"]
            raw_image_list = [img for img in images if img["key"] == str(raw_image_id)]
            if not raw_image_list:
                # The selected image doesn't have a corresponding raw image,
                # skip it
                continue
            raw_image = raw_image_list[0]
            image["uploaded_t"] = raw_image["uploaded_t"]

    # Only keep images that are selected (not raw images)
    selected_images = [
        img
        for img in images
        if not img["key"].isdigit() and img["uploaded_t"] is not None
    ]

    front_image_url = None
    ingredients_image_url = None

    # Sort images by uploaded_t descending (most recent first)
    for image in sorted(selected_images, key=lambda x: x["uploaded_t"], reverse=True):
        image_key = image["key"]
        image_rev = image["rev"]
        image_url = generate_image_url(
            code=barcode,
            image_id=f"{image_key}.{image_rev}.full",
            flavor=flavor,
        )
        if image_key.startswith("front_") and front_image_url is None:
            front_image_url = image_url
        elif image_key.startswith("ingredients_") and ingredients_image_url is None:
            ingredients_image_url = image_url
        elif front_image_url and ingredients_image_url:
            break

    results = []
    # Start with front image, then ingredients image
    if front_image_url:
        results.append(front_image_url)
    if ingredients_image_url:
        results.append(ingredients_image_url)

    return results


def convert_pydantic_model_to_google_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Google doesn't support natively OpenAPI schemas, so we convert them to
    Google `Schema` (a subset of OpenAPI)."""
    return GoogleSchema.from_json_schema(
        json_schema=GoogleJSONSchema.model_validate(schema.model_json_schema())
    ).model_dump(mode="json", exclude_none=True, exclude_unset=True)


@app.command()
def generate_dataset(
    parquet_path: Path,
    output_path: Path,
    country_filter: str | None = None,
    remove_duplicates_from_dataset: Path | None = None,
):
    """Generate a dataset file in JSONL format to be used for batch
    processing, using Gemini Batch Inference.

    Args:
        parquet_path (Path): Path to the Parquet file containing product data.
        output_path (Path): Path where to write the generated dataset file.
        country_filter (str | None): If provided, only products available in
            this country (ISO 3166-1 alpha-2 code) will be included.
        remove_duplicates_from_dataset (Path | None): If provided, path to a
            JSONL file containing previously processed products. Products
            present in this file will be excluded from the generated dataset.
            We use the "key" field in each record to identify products.
    """
    import json

    import duckdb

    processed_codes = set()
    if remove_duplicates_from_dataset:
        typer.echo(
            f"Loading previously processed products from {remove_duplicates_from_dataset}..."
        )
        with remove_duplicates_from_dataset.open("r") as f:
            for record in map(json.loads, f):
                key: str = record["key"]
                if key.startswith("food:"):
                    code = key[len("food:") :]
                    processed_codes.add(code)
                else:
                    raise ValueError(f"Unexpected key format: {key}")
        typer.echo(f"Loaded {len(processed_codes)} previously processed products.")

    # Read the parquet file using DuckDB
    conn = duckdb.connect()

    where_clauses = ["images IS NOT NULL", "length(images) > 0"]
    if country_filter:
        where_clauses.append("list_contains(food.countries_tags, 'en:canada')")

    where_clause_str = " AND ".join(where_clauses)

    full_query = f"SELECT code, images FROM '{parquet_path}' WHERE {where_clause_str}"

    typer.echo(f"Query: '{full_query}'")
    skipped = 0
    query_execute = conn.execute(full_query)

    with open(output_path, "w") as f:
        while True:
            print("Next chunk...")
            try:
                cur_chunk = query_execute.fetch_df_chunk()
            except Exception as err:
                print(err)
                break

            print(f"Chunk with {len(cur_chunk)} rows")
            for row in cur_chunk.itertuples(index=False):
                code, images = row

                if code in processed_codes:
                    skipped += 1
                    continue

                image_urls = get_image_urls(code, images, Flavor.off)

                if not image_urls:
                    continue

                parts: list[dict[str, Any]] = [{"text": DEFAULT_INSTRUCTIONS}]
                for image_url in image_urls:
                    parts.append(
                        {
                            "file_data": {
                                "file_uri": image_url,
                                "mime_type": "image/jpeg",
                            }
                        }
                    )
                record = {
                    "key": f"food:{code}",
                    "request": {
                        "contents": [
                            {
                                "parts": parts,
                                "role": "user",
                            }
                        ],
                        "generationConfig": {
                            "response_mime_type": "application/json",
                            "response_json_schema": convert_pydantic_model_to_google_schema(
                                CategoryPredictionResponseModel
                            ),
                        },
                    },
                }
                f.write(json.dumps(record) + "\n")

    typer.echo(
        f"Dataset generation completed. Wrote dataset to {output_path}. "
        f"Skipped {skipped} already processed products."
    )


async def download_image(url: str, session: aiohttp.ClientSession) -> bytes:
    """Download an image from a URL and return its content as bytes.

    Args:
        url (str): URL of the image to download.
    Returns:
        bytes: Content of the downloaded image.
    """
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()


async def download_image_from_filesystem(url: str, base_dir: Path) -> bytes:
    """Download an image from the filesystem and return its content as bytes.

    Args:
        url (str): URL of the image to download.
        base_dir (Path): Base directory where images are stored.
    Returns:
        bytes: Content of the downloaded image.
    """
    file_path = urlparse(url).path[1:]  # Remove leading '/'
    full_file_path = base_dir / file_path
    async with aiofiles.open(full_file_path, "rb") as f:
        return await f.read()


async def upload_to_gcs(
    image_url: str,
    bucket_name: str,
    blob_name: str,
    session: aiohttp.ClientSession,
    base_image_dir: Path | None = None,
) -> dict:
    """Upload data to Google Cloud Storage.
    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Name of the blob (object) in the bucket.
        data (bytes): Data to upload.
        session (aiohttp.ClientSession): HTTP session to use for downloading
            the image.
        base_image_dir (Path | None): If provided, images will be read from
            the filesystem under this base directory instead of downloading
            them from their URLs.
    Returns:
        dict: Status of the upload operation.
    """
    if base_image_dir is None:
        image_data = await download_image(image_url, session)
    else:
        image_data = await download_image_from_filesystem(image_url, base_image_dir)

    client = Storage(session=session)

    status = await client.upload(
        bucket_name,
        blob_name,
        image_data,
    )
    return status


async def upload_to_gcs_format_async(
    record: dict[str, Any],
    bucket_name: str,
    session: aiohttp.ClientSession,
    base_image_dir: Path | None = None,
) -> dict[str, Any] | None:
    parts = record["request"]["contents"][0]["parts"]
    record_key = record["key"]
    for part in parts:
        if "file_data" in part:
            file_uri = part["file_data"]["file_uri"]
            image_blob_name = f"dataset-images/{record_key}/{Path(file_uri).name}"
            # Download the image from the URL
            try:
                await upload_to_gcs(
                    image_url=file_uri,
                    bucket_name=bucket_name,
                    blob_name=image_blob_name,
                    session=session,
                    base_image_dir=base_image_dir,
                )
            except FileNotFoundError:
                return None

            part["file_data"]["file_uri"] = f"gs://{bucket_name}/{image_blob_name}"
    return record


async def upload_and_update_dataset_to_gcs_format_async(
    dataset_path: Path,
    output_path: Path,
    bucket_name: str,
    max_concurrent_uploads: int = 30,
    base_image_dir: Path | None = None,
    from_key: str | None = None,
):
    limiter = asyncio.Semaphore(max_concurrent_uploads)
    ignore = True if from_key is None else False
    missing_files = 0
    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            async with aiofiles.open(dataset_path, "r") as input_file, aiofiles.open(
                output_path, "wb"
            ) as output_file:
                async with limiter:
                    tasks = set()
                    async for line in tqdm(input_file, desc="samples"):
                        record = orjson.loads(line)
                        record_key = record["key"]
                        if from_key is not None and ignore:
                            if record_key == from_key:
                                ignore = False
                            else:
                                continue
                        task = tg.create_task(
                            upload_to_gcs_format_async(
                                record=record,
                                bucket_name=bucket_name,
                                session=session,
                                base_image_dir=base_image_dir,
                            )
                        )
                        tasks.add(task)

                        if len(tasks) >= max_concurrent_uploads:
                            for task in tasks:
                                await task
                                updated_record = task.result()
                                if updated_record is not None:
                                    await output_file.write(
                                        orjson.dumps(updated_record) + "\n".encode()
                                    )
                                else:
                                    missing_files += 1
                            tasks.clear()

    typer.echo(
        f"Upload and dataset update completed. Wrote updated dataset to {output_path}. "
        f"Missing files: {missing_files}."
    )


@app.command()
def upload_and_update_dataset_to_gcs_format(
    dataset_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the dataset file (JSONL format)",
            file_okay=True,
            dir_okay=False,
            exists=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path where to write the updated dataset file with GCS URI (JSONL format)",
            file_okay=True,
            dir_okay=False,
            exists=False,
        ),
    ],
    bucket_name: Annotated[
        str,
        typer.Option(
            ...,
            help="Name of the GCS bucket where to upload the images",
        ),
    ] = "robotoff-batch",
    max_concurrent_uploads: Annotated[
        int,
        typer.Option(
            ...,
            help="Maximum number of concurrent uploads to GCS",
        ),
    ] = 30,
    base_image_dir: Annotated[
        Path | None,
        typer.Option(
            ...,
            help="If provided, images will be read from the filesystem under this base directory instead of downloading them from their URLs.",
            file_okay=False,
            dir_okay=True,
            exists=True,
        ),
    ] = None,
    from_key: Annotated[
        str | None,
        typer.Option(..., help="If provided, ignore all samples before this key."),
    ] = None,
):
    """Upload the images present in the dataset (image URLs) to GCS, and write
    a new dataset file where the image URLs are replaced with GCS URIs.

    Args:
        dataset_path (Path): Path to the dataset file.
        output_path (Path): Path where to write the updated dataset file.
    """
    typer.echo(
        f"Uploading images from dataset {dataset_path} to GCS bucket {bucket_name}..."
    )
    typer.echo(f"Writing updated dataset to {output_path}...")
    typer.echo(f"Max concurrent uploads: {max_concurrent_uploads}...")
    typer.echo(f"Base image directory: {base_image_dir}...")
    typer.echo(f"From key: {from_key}...")
    asyncio.run(
        upload_and_update_dataset_to_gcs_format_async(
            dataset_path=dataset_path,
            output_path=output_path,
            bucket_name=bucket_name,
            max_concurrent_uploads=max_concurrent_uploads,
            base_image_dir=base_image_dir,
            from_key=from_key,
        )
    )


@app.command()
def launch_batch_job(
    run_name: Annotated[str, typer.Argument(..., help="Name of the batch job run")],
    dataset_path: Annotated[Path, typer.Option(..., help="Path to the dataset file")],
    model: Annotated[
        str, typer.Option(..., help="Model to use for the batch job")
    ] = "gemini-2.5-flash",
    location: Annotated[
        str,
        typer.Option(..., help="GCP location where to run the batch job"),
    ] = "europe-west4",
):
    from google import genai
    from google.cloud import storage
    from google.genai.types import CreateBatchJobConfig, HttpOptions

    # We upload the dataset to a GCS bucket using the Gcloud

    if model == "gemini-3-pro-preview" and location != "global":
        typer.echo(
            "Warning: only 'global' location is supported for 'gemini-3-pro-preview' model. Overriding location to 'global'."
        )
        location = "global"

    storage_client = storage.Client()
    bucket_name = "robotoff-batch"  # Replace with your bucket name
    run_dir = f"gemini-batch/{run_name}"
    input_file_blob_name = f"{run_dir}/inputs.jsonl"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(input_file_blob_name)
    blob.upload_from_filename(dataset_path)

    client = genai.Client(
        http_options=HttpOptions(api_version="v1"),
        vertexai=True,
        location=location,
    )
    output_uri = f"gs://{bucket_name}/{run_dir}"
    job = client.batches.create(
        model=model,
        src=f"gs://{bucket_name}/{input_file_blob_name}",
        config=CreateBatchJobConfig(dest=output_uri),
    )
    print(job)


@app.command()
def process_predictions(
    prediction_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the predictions JSONL file",
            file_okay=True,
            dir_okay=False,
            exists=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path where to write the processed results TSV file",
            file_okay=True,
            dir_okay=False,
            exists=False,
        ),
    ],
    limit: Annotated[
        int | None,
        typer.Option(
            ..., help="If provided, limit the number of successful records to process"
        ),
    ] = None,
):
    """Process the predictions file generated by Gemini Batch Inference,
    and create a TSV file containing the results.

    Args:
        prediction_path (Path): Path to the predictions JSONL file.
        limit (int | None): If provided, limit the number of successful records
            to process.
    """
    failed = 0
    validation_error = 0
    empty = 0
    processed = 0

    with prediction_path.open("r") as f_in, output_path.open("w", newline="") as f_out:
        csv_writer = csv.DictWriter(
            f_out,
            delimiter="\t",
            fieldnames=[
                "code",
                "url",
                "type",
                "main_category",
                "broader_categories",
                "explanation",
            ],
        )
        csv_writer.writeheader()
        for record in map(orjson.loads, f_in):
            if record["status"] != "":
                failed += 1
                continue  # Skip failed records
            content = record["response"]["candidates"][0]["content"]

            if "parts" not in content:
                empty += 1
                continue  # Skip empty responses

            output_text = content["parts"][0]["text"]

            try:
                response = CategoryPredictionResponseModel.model_validate_json(
                    output_text
                )
            except ValidationError:
                validation_error += 1
                continue  # Skip invalid responses

            processed += 1

            main_category = None if not response.categories else response.categories[0]
            remaining_categories = (
                response.categories[1:] if len(response.categories) > 1 else []
            )
            barcode = record["key"].split("food:")[1]
            csv_writer.writerow(
                {
                    "code": barcode,
                    "url": f"https://world.openfoodfacts.org/product/{barcode}",
                    "type": response.type,
                    "main_category": (
                        f"{main_category.category} ({main_category.language})"
                        if main_category
                        else ""
                    ),
                    "broader_categories": ", ".join(
                        [
                            f"{cat.category} ({cat.language})"
                            for cat in remaining_categories
                        ]
                    ),
                    "explanation": response.explanation,
                }
            )

            if limit is not None and processed >= limit:
                break


if __name__ == "__main__":
    app()
