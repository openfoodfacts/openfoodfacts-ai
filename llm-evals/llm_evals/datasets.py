import json
from pathlib import Path
from typing import Any

import typer
from ruamel.yaml import YAML


def load_yaml_dataset(dataset_path: Path):
    yaml = YAML()
    with dataset_path.open("r") as f:
        return yaml.load(f)


def save_yaml_dataset(content: dict[str, Any], dataset_path: Path):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=2, offset=0)

    with dataset_path.open("w") as f:
        yaml.dump(content, f)


def insert_sample(sample: dict[str, Any], dataset_path: Path) -> None:
    typer.echo(f"Sample:\n{json.dumps(sample, indent=2)}")
    dataset = load_yaml_dataset(dataset_path)

    typer.echo(f"Dataset cases before insertion: {len(dataset['cases'])}")

    for case in dataset["cases"]:
        if sample["name"] == case["name"]:
            raise ValueError(
                f"Duplicate sample: ID '{sample['name']}' is already in the dataset."
            )
    dataset["cases"].append(sample)
    typer.echo(f"Saving dataset with new sample to {dataset_path}")
    save_yaml_dataset(dataset, dataset_path)
