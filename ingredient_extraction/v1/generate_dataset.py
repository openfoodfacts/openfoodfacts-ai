import argparse
import json
import gzip
import logging
from pathlib import Path
from typing import Counter
from urllib.parse import urlparse

from spacy import blank

from utils import (
    PROMPT_VERSION,
    ErrorType,
    fetch_annotations,
    fetch_cached_response,
    generate_highlighted_text,
    generate_identifier,
    get_barcode_from_url,
    get_root_logger,
    parse_response,
    response_exists,
)

logger = get_root_logger(logging_level=logging.INFO)


def tokenize(nlp, text: str, offsets: list[tuple[int, int]]):
    logger.debug("offsets: %s", offsets)
    doc = nlp(text)
    spans = [
        doc.char_span(start_idx, end_idx)
        for start_idx, end_idx in sorted(set(offsets), key=lambda x: x[0])
    ]
    has_error = any(x is None for x in spans)

    if has_error:
        logger.debug("doc: '%s', offsets: %s", [t.orth_ for t in doc], offsets)
        return None, None

    spans = [x for x in spans if x is not None]
    if not spans:
        ner_tags = [0 for _ in range(len(doc))]
    else:
        span_idx = 0
        span = spans[span_idx]
        ner_tags = []
        logger.debug(f"current span: {span.start}:{span.end}")

        for i, token in enumerate(doc):
            logger.debug(f"token {i}: '{token}' (char index: {token.idx})")
            if span is not None:
                if span.start > i:
                    logger.debug("TAG: 'O'")
                    ner_tags.append(0)
                elif span.start == i:
                    logger.debug("TAG: 'B-ING'")
                    ner_tags.append(1)
                elif span.end > i:
                    logger.debug("TAG: 'I-ING'")
                    ner_tags.append(2)
                    if i == len(doc) - 1:
                        # Last token in document
                        span_idx += 1
                        span = spans[span_idx] if span_idx < len(spans) else None
                        if span:
                            logger.debug(f"current span: {span.start}:{span.end}")
                        else:
                            logger.debug("No span left")
                elif span.end == i:
                    logger.debug("TAG: 'O'")
                    ner_tags.append(0)
                    span_idx += 1
                    span = spans[span_idx] if span_idx < len(spans) else None
                    if span:
                        logger.debug(f"current span: {span.start}:{span.end}")
                    else:
                        logger.debug("No span left")
                else:
                    raise ValueError
            else:
                ner_tags.append(0)

        if span is not None:
            raise ValueError

    return [t.orth_ for t in doc], ner_tags


def generate_dataset(urls: list[str], output_dir: Path, test_split_set: set[str]):
    errors: list[ErrorType] = []
    tokenization_errors = 0
    dataset = []
    highlighted_text_by_split = {"train": "", "test": ""}
    split_counts = {"train": 0, "test": 0}
    count = 0

    nlp = blank("en")
    annotations = fetch_annotations()

    for url in urls:
        if url.endswith(".jpg"):
            url = url.replace(".jpg", ".json")
        if url.startswith("https://images."):
            url = url.replace("https://images.", "https://static.")

        image_url = url.replace(".json", ".jpg")
        barcode = get_barcode_from_url(url)
        image_id = Path(urlparse(url).path).stem
        id_ = generate_identifier(barcode, image_id, PROMPT_VERSION)

        if not response_exists(id_):
            logger.debug("Response for id %s does not exist", id_)
            continue

        count += 1
        response = fetch_cached_response(id_)
        full_text = response["data"]["text"]
        text_response = response["choices"][0]["message"]["content"]

        parsed_json, error = parse_response(text_response, id_, full_text)
        if id_ in annotations:
            annotation = annotations[id_]
            action = annotation["action"]
            if action == "a":
                # Set error to False
                error = None
            elif action == "r":
                error = (
                    "manually_rejected",
                    id_,
                    "annotation has been manually rejected",
                )
            elif action == "u":
                error = None
                parsed_json = annotation["updated_json"]

        if error is not None:
            errors.append(error)
        else:
            entity_offsets = [(x["start_idx"], x["end_idx"]) for x in parsed_json]
            tokens, ner_tags = tokenize(nlp, full_text, entity_offsets)
            tokenization_errors += int(bool(tokens is None))
            split = "test" if id_ in test_split_set else "train"
            split_counts[split] += 1
            if tokens:
                dataset.append(
                    {
                        "text": full_text,
                        "marked_text": generate_highlighted_text(
                            full_text,
                            entity_offsets,
                            html_escape=False,
                            start_token="<b>",
                            end_token="</b>",
                        ),
                        "annotations": parsed_json,
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                        "meta": {
                            "barcode": barcode,
                            "image_id": image_id,
                            "id": id_,
                            "url": url,
                            "in_test_split": id_ in test_split_set,
                        },
                    }
                )
            highlighted_text_by_split[split] += (
                "<p>"
                + generate_highlighted_text(
                    full_text,
                    entity_offsets,
                    html_escape=True,
                    start_token="<b>",
                    end_token="</b>",
                )
                + f'</br>{id_}, <a href="{image_url}">{image_url}</a>'
                + ("" if tokens else " tokenization error")
                + "</p>"
            )

    error_counts = Counter(x[0] for x in errors)
    logger.info("Number of items: %d", count)

    logger.info("# Error counts")
    logger.info("  Tokenization errors: %d", tokenization_errors)
    for error_type, count in error_counts.most_common():
        logger.info("  %s: %d", error_type, count)

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
        "--url-list-path",
        type=Path,
        help="path of text file containing OCR URL to use for ingredient detection "
        "(one line per URL). Takes precedence over --url parameter",
        default="image_urls.txt",
    )
    parser.add_argument("--test-split-path", type=Path)
    args = parser.parse_args()
    urls = args.url_list_path.read_text().splitlines()

    test_split_set = (
        set()
        if not args.test_split_path
        else set(args.test_split_path.read_text().splitlines())
    )
    OUTPUT_DIR = Path(__file__).parent
    generate_dataset(urls, OUTPUT_DIR, test_split_set)
