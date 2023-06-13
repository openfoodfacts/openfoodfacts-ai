import argparse
import json
import gzip
import logging
from pathlib import Path

from utils import (
    fetch_annotations,
    generate_highlighted_text,
    get_root_logger,
    jsonl_iter,
    tokenize,
)

logger = get_root_logger(logging_level=logging.INFO)


def generate_dataset(items: list[dict], output_dir: Path):
    tokenization_errors = 0
    dataset = []
    highlighted_text_by_split = {"train": "", "test": ""}
    split_counts = {"train": 0, "test": 0}
    count = 0
    rejected = 0
    updated = 0

    annotations = fetch_annotations()

    for item in items:
        id_ = item["meta"]["id"]
        count += 1
        full_text = item["text"]
        entity_offsets = item["offsets"]

        if id_ in annotations:
            annotation = annotations[id_]
            action = annotation["action"]
            if action == "r":
                rejected += 1
                continue
            elif action == "u":
                entity_offsets = annotation["updated_offsets"]
                updated += 1

        entity_offsets = list(map(tuple, entity_offsets))
        entity_offsets = [x for x in entity_offsets if x[0] < x[1]]
        tokens, ner_tags = tokenize(full_text, entity_offsets)
        split = "test" if item["meta"]["in_test_split"] else "train"
        split_counts[split] += 1

        if tokens:
            dataset.append(
                {
                    "text": full_text,
                    "marked_text": generate_highlighted_text(full_text, entity_offsets),
                    "offsets": entity_offsets,
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "meta": item["meta"],
                }
            )
        else:
            tokenization_errors += 1

        image_url = item["meta"]["url"].replace(".json", ".jpg")
        highlighted_text_by_split[split] += (
            "<p>"
            + generate_highlighted_text(
                full_text,
                entity_offsets,
                html_escape=True,
                start_token="<mark>",
                end_token="</mark>",
            )
            + f'</br>{id_}, <a href="{image_url}">{image_url}</a>'
            + ("" if tokens else " tokenization error")
            + "</p>"
        )

    logger.info("Number of items: %d", count)
    logger.info(f"{rejected=}, {updated=}")
    logger.info("  Tokenization errors: %d", tokenization_errors)

    with gzip.open(output_dir / f"dataset.jsonl.gz", "wt") as f:
        f.write("\n".join(map(json.dumps, dataset)))

    for split_name in ("train", "test"):
        logger.info(f"[split:{split_name}] valid: items: %s", split_counts[split_name])

        (output_dir / f"dataset_{split_name}.html").write_text(
            f"<html><body>{highlighted_text_by_split[split_name]}</body></html>"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default="dataset_base.jsonl.gz",
    )
    args = parser.parse_args()
    OUTPUT_DIR = Path(__file__).parent
    generate_dataset(jsonl_iter(args.dataset_path), OUTPUT_DIR)
