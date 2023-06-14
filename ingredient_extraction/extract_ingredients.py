import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import tqdm

from robotoff.off import get_barcode_from_url
from robotoff.prediction.ocr.core import get_ocr_result
from robotoff.utils import http_session

LOGGING_LEVEL = logging.INFO
logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
handler.setFormatter(formatter)
handler.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)


RESPONSE_SAVE_DIR = Path(__file__).parent / "responses"
RESPONSE_SAVE_DIR.mkdir(exist_ok=True)


OPENAI_API_KEY = os.environ["OPENAPI_KEY"]

PROMPT_VERSION = "4"
PROMPTS = {
    "4": """Extract ingredient lists from the following texts. The ingredient list should start with the first ingredient and end with the last ingredient. It should not include allergy, label or origin information.
The output format must be a single JSON list containing one element per ingredient list. If there are ingredients in several languages, the output JSON list should contain as many elements as detected languages. Each element should have two fields:
- a "text" field containing the detected ingredient list. The text should be a substring of the original text, you must not alter the original text.
- a  "lang" field containing the detected language of the ingredient list.
Don't output anything else than the expected JSON list.""",
    # With prompt 5, we output more frequently empty list, so we didn't keep it
#     "5": """Extract ingredient lists from the following text. The ingredient list should start with the first ingredient and end with the last ingredient. It should not include allergy, label or origin information.
# The output format must be a single JSON list containing one element per ingredient list. Output an empty list if there are no ingredient lists. If there are ingredients in several languages, the output JSON list should contain as many elements as detected languages. Each element should have two fields:
# - a "text" field containing the detected ingredient list. The text should be a substring of the original text, you must not alter the original text.
# - a  "lang" field containing the detected language of the ingredient list.
# Don't output anything else than the expected JSON list.""",
}

PROMPT = PROMPTS[PROMPT_VERSION]


class ContextLengthExceededException(Exception):
    pass


def send_request(text: str) -> dict:
    user_message = f"{PROMPT}\n\n'''{text}'''"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ChatGPT, a large language model trained by OpenAI. "
                    "Only generate responses in JSON format. The output JSON must be minified."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    }

    while True:
        r = http_session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        )

        if r.status_code == 429 or str(r.status_code).startswith("5"):
            if r.status_code == 429:
                logger.info("Rate limit reached, sleeping for 1 min")
                time.sleep(60)
            else:
                logger.info("%d HTTP error, sleeping for 5s", r.status_code)
                time.sleep(5)
        else:
            break

    if (
        r.status_code == 400
        and r.json()["error"].get("code") == "context_length_exceeded"
    ):
        raise ContextLengthExceededException()

    r.raise_for_status()
    return r.json()


def save_response(identifier: str, response: dict):
    file_path = RESPONSE_SAVE_DIR / f"{identifier}.json"
    logger.debug("Saving OpenAI response in %s", file_path)

    with file_path.open("w") as f:
        json.dump(response, f, ensure_ascii=False)


def response_exists(identifier: str):
    return (RESPONSE_SAVE_DIR / f"{identifier}.json").is_file()


def fetch_cached_response(identifier: str):
    file_path = RESPONSE_SAVE_DIR / f"{identifier}.json"
    if file_path.is_file():
        return json.loads(file_path.read_text())


def extract_ingredients(identifier: str, text: str) -> Optional[list]:
    if (response := fetch_cached_response(identifier)) is None:
        try:
            response = send_request(text)
        except ContextLengthExceededException:
            logger.info(f"Context length exception for {identifier}")
            return None

        response["data"] = {
            "prompt": PROMPT,
            "text": text,
        }
        save_response(identifier, response)
    else:
        logger.debug("Cached response used for %s", identifier)

    usage = response["usage"]
    logger.debug(
        "Usage: completion tokens: %s, prompt tokens: %s, total tokens: %s",
        usage["completion_tokens"],
        usage["prompt_tokens"],
        usage["total_tokens"],
    )
    choice = response["choices"][0]
    text_response = choice["message"]["content"]

    logger.debug("Trying to parse JSON %s", text_response)
    try:
        parsed_json = json.loads(text_response)
    except json.JSONDecodeError:
        logger.debug("invalid JSON for identifier %s: %s", identifier, text_response)
        return None

    if not isinstance(parsed_json, list):
        logger.debug("JSON is not a list: %s", parsed_json)
        return None

    for item in parsed_json:
        if not isinstance(item, dict):
            logger.debug("item %s is not a dict", item)
            return None
        elif (
            "text" not in item
            or "lang" not in item
            or not isinstance(item["text"], str)
            or not isinstance(item["lang"], str)
        ):
            logger.debug("invalid key/values for item %s", item)
            return None

    return parsed_json


def extract_ingredient_from_ocr_url(url: str) -> bool:
    if url.endswith(".jpg"):
        url = url.replace(".jpg", ".json")
    if url.startswith("https://images."):
        url = url.replace("https://images.", "https://static.")

    logger.debug("Extracting ingredients from %s", url)
    barcode = get_barcode_from_url(url)

    if barcode is None:
        raise ValueError("invalid OCR url: %s", url)

    image_id = Path(urlparse(url).path).stem
    id_ = f"{barcode}_{image_id}_prompt_version-{PROMPT_VERSION}"

    if not response_exists(id_):
        ocr_result = get_ocr_result(url, http_session)
        if not ocr_result:
            return False
        full_text = ocr_result.get_full_text()
        if not full_text:
            logger.debug("Empty text for ID %s", id_)
            return False
        full_text = full_text.replace("|", " ")
    else:
        full_text = fetch_cached_response(id_)["data"]["text"]

    logger.debug("Ingredient text: %s", full_text.replace("\n", "<LF>"))
    extracted_ingredients = extract_ingredients(id_, full_text)

    if extracted_ingredients is None:
        return False

    logger.debug(
        "Extracted ingredients:\n%s",
        json.dumps(extracted_ingredients, indent=4, ensure_ascii=False),
    )

    valid = True
    for i, extracted_ingredient in enumerate(extracted_ingredients):
        if "text" not in extracted_ingredient or "lang" not in extracted_ingredient:
            logger.debug("invalid generated JSON schema for ingredient #%s", i + 1)
            valid = False
        else:
            extracted_ingredient_text = extracted_ingredient["text"]
            if extracted_ingredient_text not in full_text:
                logger.debug(
                    "Extracted ingredient #%s is not a substring of original text:\n%s",
                    i + 1,
                    extracted_ingredient_text,
                )
                valid = False
            else:
                logger.debug("Ingredient #%s is valid", i + 1)

    return valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url_list_path",
        help="path of text file containing OCR URL to use for ingredient detection "
        "(one line per URL). Takes precedence over --url parameter",
    )
    args = parser.parse_args()

    if args.url_list_path:
        urls = Path(args.url_list_path).read_text().splitlines()
    else:
        urls = [args.url]

    url_iter = tqdm.tqdm(urls)
    valid_count = 0
    for url in url_iter:
        is_valid = extract_ingredient_from_ocr_url(url)
        valid_count += int(is_valid)
        url_iter.postfix = f"valid={valid_count}"
