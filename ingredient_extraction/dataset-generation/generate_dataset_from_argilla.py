import argparse
import gzip
import json
import logging
from pathlib import Path

import argilla as rg
from argilla._constants import DEFAULT_API_KEY
from argilla.client.datasets import DatasetForTokenClassification

from utils import generate_highlighted_text, get_root_logger, tokenize

logger = get_root_logger(logging_level=logging.INFO)


def generate_dataset(ds: DatasetForTokenClassification, output_dir: Path):
    tokenization_errors = 0
    dataset = []
    highlighted_text_by_split = {"train": "", "test": ""}
    split_counts = {"train": 0, "test": 0}
    count = 0
    rejected = 0
    updated = 0

    for item in ds:
        id_ = item.id
        item.metadata["id"] = id_
        count += 1
        full_text = item.text

        if item.annotation is not None:
            # A human annotation was provided
            entity_offsets = [(ent[1], ent[2]) for ent in item.annotation]
        else:
            entity_offsets = [(ent[1], ent[2]) for ent in item.prediction]

        entity_offsets = [x for x in entity_offsets if x[0] < x[1]]
        tokens, ner_tags = tokenize(full_text, entity_offsets)
        split = "test" if item.metadata["in_test_split"] else "train"

        if tokens:
            split_counts[split] += 1
            dataset.append(
                {
                    "text": full_text,
                    "marked_text": generate_highlighted_text(full_text, entity_offsets),
                    "offsets": entity_offsets,
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "meta": item.metadata,
                }
            )
        else:
            tokenization_errors += 1

        image_url = item.metadata["url"].replace(".json", ".jpg")
        highlighted_text_by_split[split] += (
            "<p>"
            + generate_highlighted_text(
                full_text, entity_offsets, html_escape=True, mark_token="mark"
            )
            + f'</br>{id_}, <a href="{image_url}">{image_url}</a>'
            + ("" if tokens else " tokenization error")
            + "</p>"
        )

    logger.info("Number of items: %d", count)
    logger.info(f"{rejected=}, {updated=}")
    logger.info("  Tokenization errors: %d", tokenization_errors)

    with gzip.open(output_dir / "dataset.jsonl.gz", "wt") as f:
        f.write("\n".join(map(json.dumps, dataset)))

    for split_name in ("train", "test"):
        logger.info(f"[split:{split_name}] valid: items: %s", split_counts[split_name])

        (output_dir / f"dataset_{split_name}.html").write_text(
            f"<html><body>{highlighted_text_by_split[split_name]}</body></html>"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rg.init(api_url="http://localhost:6900", api_key=DEFAULT_API_KEY)
    ds = rg.load("ingredient-detection-ner")
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    generate_dataset(ds, output_dir=output_dir)
