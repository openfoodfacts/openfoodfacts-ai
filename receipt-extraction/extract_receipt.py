import json
from pathlib import Path
from typing import Optional

import typer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


class ReceiptItem(BaseModel):
    """Data class for storing text attributes of a receipt item."""

    name: str | None = Field(description="The name of the item.")
    unit_price: str | None = Field(
        description="The unit price of the item. It should be None if the item is sold by weight."
    )
    price: str | None = Field(
        description="The price of the item, for all units. "
        "If the item is sold by weight, this should be the total price."
    )
    discount: str | None = Field(
        description="The discount applied to the item. It should be None if no discount was applied."
    )
    price_per_kg: str | None = Field(
        description="The price per kg of the item. It should be None if the item is not sold by weight."
    )
    quantity: str | None = Field(description="The number of items bought.")
    weight: str | None = Field(
        description="The weight of the item. It should be None if the item is not sold by weight."
    )


# Desired output structure
class Receipt(BaseModel):
    """Data class for storing text attributes of a shop receipt."""

    datetime: str | None = Field(
        description="The datetime at which the receipt was issued."
    )
    shop_name: str | None = Field(description="The name of the shop.")
    address: str | None = Field(description="The full address of the shop.")
    total: str | None = Field(description="The total amount paid, after reductions.")
    total_before_reduction: str | None = Field(
        description="The total amount paid, before reductions."
    )
    items: list[ReceiptItem] | None = Field(
        description="The list of items bought on the receipt."
    )
    item_count: int | None = Field(
        description="The number of items bought, as displayed on the receipt."
    )
    currency: str | None = Field(
        description="The currency used in the receipt, uses ISO 4217 code."
    )


PROMPT = """Use the attached Receipt image to extract data from it and store into the
provided data class. The receipt was issued either in a shop or through an online shopping website."""


def main(
    input_dir: Path = typer.Argument(
        ..., help="The directory containing the images of the receipts."
    ),
    output_dir: Path = typer.Argument(
        ..., help="The directory to store the extracted data."
    ),
    max_new_tokens: int = typer.Option(
        4096, help="The maximum number of tokens to generate."
    ),
    max_images: Optional[int] = typer.Option(
        None, help="The maximum number of images to process."
    ),
):
    """Extracts data from images of receipts using GPT-4 OpenAI model.

    The OPENAI_API_KEY environment variable must be set to use the OpenAI API.
    """
    typer.echo("Running receipt extraction script.")
    image_documents = SimpleDirectoryReader(input_dir).load_data()
    gpt_4o = OpenAIMultiModal(model="gpt-4o", max_new_tokens=max_new_tokens)
    mmllm = MultiModalLLMCompletionProgram.from_defaults(
        output_cls=Receipt,
        prompt_template_str=PROMPT,
        multi_modal_llm=gpt_4o,
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    skipped = 0
    extracted = 0
    for i, image_document in enumerate(image_documents):
        filename = Path(image_document.image_path).name
        output_path = output_dir / f"{filename}.json"

        if output_path.exists():
            typer.echo(f"Skipping {image_document.image_path}")
            skipped += 1
            continue

        if max_images is not None and (i - skipped) >= max_images:
            typer.echo(f"Reached max_images limit of {max_images}.")
            break

        typer.echo(f"Extracting data from {image_document.image_path}")
        receipt = mmllm(image_documents=[image_document])
        extracted += 1
        data = {
            "receipt": receipt.model_dump(),
            "model": "gpt-4o",
            "image_name": filename,
        }

        with output_path.open("w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    typer.echo(f"Extracted: {extracted}, Skipped: {skipped}")


if __name__ == "__main__":
    typer.run(main)
