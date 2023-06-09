import html
import json
from pathlib import Path
import re
from typing import Optional, Union
import logging
from urllib.parse import urlparse

from db import Annotation

logger = logging.getLogger(__name__)

ErrorType = tuple[str, str, str]


RESPONSE_SAVE_DIR = Path(__file__).parent / "responses"
PROMPT_VERSION = "4"


def sanitize_json(
    parsed_json: Union[list, dict], identifier: str, full_text: str
) -> tuple[Optional[list], Optional[ErrorType]]:
    if not isinstance(parsed_json, list):
        return (
            None,
            (
                "json_not_list",
                identifier,
                f"invalid JSON: {parsed_json}",
            ),
        )

    seen_languages = set()
    span_offsets: list[tuple[int, int]] = []
    for i, item in enumerate(parsed_json):
        if not isinstance(item, dict):
            return (
                None,
                (
                    "json_item_not_dict",
                    identifier,
                    f"At least of the JSON element is not a dict: {parsed_json}",
                ),
            )
        elif (
            "text" not in item
            or "lang" not in item
            or not isinstance(item["text"], str)
            or not isinstance(item["lang"], str)
        ):
            return (
                parsed_json,
                (
                    "invalid_key_values",
                    identifier,
                    f"At least of the JSON element has invalid key-value pairs: {parsed_json}",
                ),
            )

        lang = item["lang"]
        if lang in seen_languages:
            return (
                parsed_json,
                (
                    "duplicated_lang",
                    identifier,
                    f"duplicated lang detected: {parsed_json}",
                ),
            )

        seen_languages.add(lang)
        extracted_ingredient_text = item["text"].strip(" ,.")
        item["text"] = extracted_ingredient_text
        if extracted_ingredient_text not in full_text:
            return (
                parsed_json,
                (
                    "ingredient_no_match",
                    identifier,
                    f"Extracted ingredient #{i + 1} is not a substring of original text:\n{full_text}",
                ),
            )
        lang = item.pop("lang").lower()

        if "/" in lang:
            langs = [l.strip() for l in lang.split("/")]
        else:
            langs = [lang]

        for lang in langs:
            if len(lang) != 2:
                return (
                    parsed_json,
                    (
                        "invalid_lang",
                        identifier,
                        f"invalid lang: {lang}",
                    ),
                )

        start_idx, end_idx = find_span_offsets(full_text, extracted_ingredient_text)
        item["start_idx"] = start_idx
        item["end_idx"] = end_idx

        logger.debug(f"{start_idx=}, {end_idx=}, {span_offsets=}")
        for existing_offset in span_offsets:
            if start_idx < existing_offset[1] and end_idx > existing_offset[0]:
                return (
                    parsed_json,
                    (
                        "conflicting_offsets",
                        identifier,
                        f"conflicting offsets: ({start_idx}, {end_idx}) with {existing_offset}",
                    ),
                )
        span_offsets.append((start_idx, end_idx))

    return parsed_json, None


def response_exists(identifier: str):
    return (RESPONSE_SAVE_DIR / f"{identifier}.json").is_file()


def fetch_cached_response(identifier: str):
    file_path = RESPONSE_SAVE_DIR / f"{identifier}.json"
    if file_path.is_file():
        return json.loads(file_path.read_text())


def find_span_offsets(text: str, substring: str):
    start_idx = text.find(substring)
    if start_idx == -1:
        raise ValueError()
    return start_idx, start_idx + len(substring)


def parse_response(
    text_response: str, identifier: str, full_text: str
) -> tuple[Optional[list], Optional[ErrorType]]:
    logger.debug("Trying to parse JSON %s", text_response)
    try:
        parsed_json = json.loads(text_response)
    except json.JSONDecodeError:
        return (
            None,
            (
                "invalid_json",
                identifier,
                f"invalid JSON for {text_response}",
            ),
        )
    return sanitize_json(parsed_json, identifier, full_text)


def get_barcode_from_path(path: str) -> Optional[str]:
    barcode = ""

    for parent in Path(path).parents:
        if parent.name.isdigit():
            barcode = parent.name + barcode
        else:
            break

    return barcode or None


def get_barcode_from_url(url: str) -> Optional[str]:
    url_path = urlparse(url).path
    return get_barcode_from_path(url_path)


BARCODE_PATH_REGEX = re.compile(r"^(...)(...)(...)(.*)$")


def generate_image_path(barcode: str, image_id: str) -> str:
    if not barcode.isdigit():
        raise ValueError("unknown barcode format: {}".format(barcode))

    match = BARCODE_PATH_REGEX.fullmatch(barcode)

    if match:
        splitted_barcode = [x for x in match.groups() if x]
    else:
        splitted_barcode = [barcode]
    return "/{}/{}.jpg".format("/".join(splitted_barcode), image_id)


def get_root_logger(logging_level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging_level)
    logger.addHandler(handler)
    return logger


def generate_identifier(barcode: str, image_id: str, prompt_version: str):
    return f"{barcode}_{image_id}_prompt_version-{prompt_version}"


def get_image_url_from_identifier(identifier: str):
    barcode, image_id, *_ = identifier.split("_")
    image_path = generate_image_path(barcode, image_id)
    return f"https://static.openfoodfacts.org/images/products{image_path}"


def fetch_annotations() -> dict[str, dict]:
    return {
        annotation["identifier"]: annotation
        for annotation in Annotation.select().dicts()
    }


def generate_highlighted_text(
    text: str,
    offsets: list[tuple[int, int]],
    html_escape: bool,
    start_token: str,
    end_token: str,
) -> str:
    highlighted_text = []
    previous_idx = 0
    escape_func = (lambda x: x) if html_escape is False else html.escape
    for start_idx, end_idx in offsets:
        highlighted_text.append(
            escape_func(text[previous_idx:start_idx])
            + start_token
            + escape_func(text[start_idx:end_idx])
            + end_token
        )
        previous_idx = end_idx
    highlighted_text.append(escape_func(text[previous_idx:]))
    return "".join(highlighted_text)
