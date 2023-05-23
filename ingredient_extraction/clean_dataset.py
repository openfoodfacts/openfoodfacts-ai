import argparse
import json
from pathlib import Path
import re
from typing import Optional
from urllib.parse import urlparse

import typer
from db import create_annotation
from utils import (
    PROMPT_VERSION,
    fetch_annotations,
    fetch_cached_response,
    generate_identifier,
    get_barcode_from_url,
    get_image_url_from_identifier,
    get_root_logger,
    parse_response,
    response_exists,
)
from rich.prompt import Prompt, Confirm
from rich import print

logger = get_root_logger()


def find_by_pattern(pattern: str, dataset: str):
    pass


def analyze_conflicting_offsets(urls: list[str]):
    pass


UPDATED_PAYLOAD_PATH = Path("/tmp/updated_payload.json")


def annotate(id_: str, full_text: str, parsed_json: Optional[list]):
    print(f"Image URL: {get_image_url_from_identifier(id_)}")
    print(f"ID: {id_}")
    print(f"Full text: ```{full_text}```")
    print(parsed_json)
    action = Prompt.ask("Action (a/r/u/s)", choices=["a", "r", "u", "s"])
    if action == "s":
        return
    if action in ("a", "r"):
        create_annotation(id_, action=action)
        print("Created :cross:")
    else:
        while True:
            UPDATED_PAYLOAD_PATH.unlink()
            UPDATED_PAYLOAD_PATH.touch()
            if isinstance(parsed_json, list) and all(
                "text" in item and "langs" in item for item in parsed_json
            ):
                UPDATED_PAYLOAD_PATH.write_text(
                    json.dumps(
                        [
                            {"text": item["text"], "lang": "/".join(item["langs"])}
                            for item in parsed_json
                        ],
                        indent=4,
                        ensure_ascii=False,
                    )
                )

            Confirm.ask(f"Is the payload in {UPDATED_PAYLOAD_PATH} ready to be saved")
            updated_json_str = UPDATED_PAYLOAD_PATH.read_text()
            updated_json, error = parse_response(updated_json_str, id_, full_text)
            if error is None:
                create_annotation(id_, action, updated_json)
                print("Created :cross:")
                break
            print(f"Error `{error[0]}`:\n{error[2]}")


def matches_pattern(parsed_json: list[dict], pattern: re.Pattern) -> bool:
    for item in parsed_json:
        if pattern.search(item["text"]) is not None:
            return True
    return False


def has_single_word_ingredient(parsed_json: list[dict]) -> bool:
    for item in parsed_json:
        words = item["text"].split(" ")
        if len(words) == 1:
            return True
    return False


def run(
    url_path: Path = typer.Argument(
        ...,
        help="path of text file containing OCR URL to use for ingredient detection (one line per URL)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    pattern: Optional[str] = typer.Option(None),
    single_ingredient: bool = False,
    error_type: Optional[str] = typer.Option(
        None, help="analyze items with a specific error type"
    ),
):
    urls = url_path.read_text().splitlines()
    existing_annotations = fetch_annotations()
    pattern = re.compile(pattern, re.I) if pattern else None

    logger.info("pattern: %s", pattern)

    for url in urls:
        barcode = get_barcode_from_url(url)
        image_id = Path(urlparse(url).path).stem
        id_ = generate_identifier(barcode, image_id, PROMPT_VERSION)
        if id_ in existing_annotations:
            logger.debug("Skipping item %s (already manually annotated)", id_)
            continue

        if not response_exists(id_):
            logger.debug("Response for id %s does not exist", id_)
            continue

        response = fetch_cached_response(id_)
        full_text = response["data"]["text"]
        openai_response = response["choices"][0]["message"]["content"]
        parsed_json, error = parse_response(openai_response, id_, full_text)

        if error_type and error is not None and error[0] == error_type:
            print(f"Error with type: {error_type}")
            if parsed_json is None:
                print(openai_response)
            annotate(id_, full_text, parsed_json)

        if error is None:
            if pattern and matches_pattern(parsed_json, pattern):
                print(f"item: {id_}, pattern matched")
                annotate(id_, full_text, parsed_json)
            elif single_ingredient and has_single_word_ingredient(parsed_json):
                print(f"item: {id_}, has single ingredient")
                annotate(id_, full_text, parsed_json)


if __name__ == "__main__":
    typer.run(run)
