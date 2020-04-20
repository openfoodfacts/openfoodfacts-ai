import click
import pandas as pd
from pathlib import Path
from pprint import pprint
from utils import load_dataset
from ingredients import format_ingredients

from paths import model_name_from_path, METRICS_DF_FILENAME


@click.group()
def cli():
    pass


@cli.command()
@click.option("--path", "-p", help="Path to dataset.")
@click.option("--item-id", help="Item id to review.")
def review(path, item_id):
    item = load_dataset(path, as_dict=True)[item_id]

    print_with_newline(f'Id : {item["_id"]}')
    print_with_newline(f'Original   : {item["original"]}')
    print_with_newline(f'Correct    : {item["correct"]}')
    if "prediction" in item:
        print_with_newline(f'Prediction : {item["prediction"]}')
    if "tags" in item:
        print_with_newline(f'Tags     : {item["tags"]}')

    original_ingredients = format_ingredients(item["original"])
    correct_ingredients = format_ingredients(item["correct"])
    predicted_ingredients = format_ingredients(item["prediction"])

    print_with_newline(f"Original ingredients :")
    print(sorted(original_ingredients))
    print_with_newline(f"Correct ingredients : ")
    print(sorted(correct_ingredients))
    print_with_newline(f"Predicted ingredients : ")
    print(sorted(predicted_ingredients))

    print("\n")

    print_with_newline(f"Not original, correct")
    print(sorted(correct_ingredients - original_ingredients))
    print_with_newline(f"Not original, predicted")
    print(sorted(predicted_ingredients - original_ingredients))
    print_with_newline(f"Not original, correct, predicted")
    print(sorted((correct_ingredients & predicted_ingredients) - original_ingredients))
    print_with_newline(f"Original, correct, not predicted")
    print(sorted((original_ingredients & correct_ingredients) - predicted_ingredients))


@cli.command()
@click.option("--path", "-p", "paths", multiple=True, help="Path to dataset.")
def suggest(paths):
    df = pd.DataFrame(
        [
            df["ingr_precision"]
            for df in [
                pd.read_csv(Path(path) / METRICS_DF_FILENAME, index_col="_id")
                for path in paths
            ]
        ]
    )
    print(df.nunique(dropna=False).sort_values(ascending=False).head(20))


@cli.command()
@click.option("--path", "-p", "paths", multiple=True, help="Path to dataset.")
@click.option("--item-id", help="Item id to review.")
def compare(paths, item_id):
    if not paths:
        raise ValueError("Need at least 1 path.")
    items = [load_dataset(path, as_dict=True)[item_id] for path in paths]
    first_item = items[0]

    model_names = [model_name_from_path(path) for path in paths]

    print(f'Original :\n{first_item["original"]}')
    print(f'Correct :\n{first_item["correct"]}')
    for item, model_name in zip(items, model_names):
        print(f'{model_name} :\n{item["prediction"]}')


def print_with_newline(*args, **kwargs):
    print()
    return print(*args, **kwargs)


if __name__ == "__main__":
    cli()
