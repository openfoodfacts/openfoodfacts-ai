"""This script allow to match a product category in English to the Open Food Facts category taxonomy.

A LVLM (Large Visual Language Model) is used to perform the matching, by providing a reduced version of the full taxonomy as input.
"""

# /// script
# dependencies = [
#     "openfoodfacts",
#     "typer",
#     "diskcache",
#     "pydantic",
#     "pydantic-ai",
# ]
# ///

from pathlib import Path
from typing import Annotated

import diskcache
import typer
from openfoodfacts.taxonomy import get_taxonomy
from pydantic import BaseModel, Field
from pydantic_ai import Agent

app = typer.Typer()
cache = diskcache.Cache(directory=Path(__file__).parent / ".diskcache")


INSTRUCTIONS_TEMPLATE = """Your task is to match a category (provided as input) to the Open Food Facts category taxonomy.
The category is in free-form text, and your goal is to find the best match among all items in the taxonomy.
Both the input category and a minified version of the category taxonomy are provided below.
All items of the category taxonomy are provided, with their ID, english and french translations (if available).
---
Taxonomy:\n
{taxonomy}
Input: '{input}'
"""


class Output(BaseModel):
    id: str | None = Field(
        description="the ID of the item in the taxonomy that matched the input. Must be null if no match could be found."
    )
    comments: str | None = Field(
        description="An optional field to provide additional details about the matching process."
    )


@cache.memoize()
def generate_taxonomy_input() -> str:
    taxonomy = get_taxonomy("category")

    taxonomy_str_items = []
    for node in taxonomy.iter_nodes():
        node_str = f"id={node.id}"
        for lang in ("en", "fr"):
            if node.synonyms.get(lang, []):
                synonym_str = ",".join(syn for syn in node.synonyms[lang])
                node_str += f"; {lang}={synonym_str}"
        taxonomy_str_items.append(node_str)
    return "\n".join(taxonomy_str_items)


@cache.memoize()
def extract(value: str, model: str):
    agent = Agent(model=model)
    taxonomy_input = generate_taxonomy_input()
    user_prompt = INSTRUCTIONS_TEMPLATE.format(taxonomy=taxonomy_input, input=value)
    response = agent.run_sync(user_prompt=user_prompt, output_type=Output)
    return response.output


@cache.memoize()
def extract_multiple(values: list[str], model: str):
    agent = Agent(model=model)
    taxonomy_input = generate_taxonomy_input()
    user_prompt = INSTRUCTIONS_TEMPLATE.format(
        taxonomy=taxonomy_input, input="; ".join(values)
    )
    response = agent.run_sync(user_prompt=user_prompt, output_type=Output)
    return response.output


@app.command()
def match_taxonomy(
    value: str,
    model: Annotated[str, typer.Option()] = "openrouter:deepseek/deepseek-v4-flash",
):
    typer.echo(f"Value to match: '{value}'")
    response = extract(value=value, model=model)
    print(response)


@app.command()
def match_taxonomy_multiple(
    values: list[str],
    model: Annotated[str, typer.Option()] = "openrouter:deepseek/deepseek-v4-flash",
):
    typer.echo(f"Values to match: '{values}'")
    response = extract_multiple(values=values, model=model)
    print(response)


if __name__ == "__main__":
    app()
