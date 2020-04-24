import click
import pandas as pd
from pathlib import Path
from pprint import pprint
from collections import Counter

from utils import load_dataset
from ingredients import normalize_item_ingredients, tokenize_ingredients

from paths import model_name_from_path, METRICS_DF_FILENAME


@click.group()
def cli():
    pass


@cli.command()
@click.option("--path", "-p", help="Path to dataset.")
@click.option("--item-id", help="Item id to review.")
def review(path: str, item_id: str) -> None:
    item = load_dataset(path, as_dict=True)[item_id]
    item = normalize_item_ingredients(item)

    print_with_newline(f'Id : {item["_id"]}')
    print_with_newline(f'Original   : {item["original"]}')
    print_with_newline(f'Correct    : {item["correct"]}')
    if "prediction" in item:
        print_with_newline(f'Prediction : {item["prediction"]}')
    if "tags" in item:
        print_with_newline(f'Tags     : {item["tags"]}')

    original_ingredients = Counter(
        tokenize_ingredients(item["original"], remove_plural=True)
    )
    correct_ingredients = Counter(
        tokenize_ingredients(item["correct"], remove_plural=True)
    )
    predicted_ingredients = Counter(
        tokenize_ingredients(item["prediction"], remove_plural=True)
    )

    print_with_newline(f"Original ingredients :")
    print(original_ingredients)
    print_with_newline(f"Correct ingredients : ")
    print(correct_ingredients)
    print_with_newline(f"Predicted ingredients : ")
    print(predicted_ingredients)

    print("\n")

    print_with_newline(f"Not original, correct")
    print(correct_ingredients - original_ingredients)
    print_with_newline(f"Not original, predicted")
    print(predicted_ingredients - original_ingredients)
    print_with_newline(f"Not original, correct, predicted")
    print((correct_ingredients & predicted_ingredients) - original_ingredients)
    print_with_newline(f"Original, correct, not predicted")
    print((original_ingredients & correct_ingredients) - predicted_ingredients)


def print_with_newline(*args, **kwargs):
    print()
    return print(*args, **kwargs)


if __name__ == "__main__":
    cli()
