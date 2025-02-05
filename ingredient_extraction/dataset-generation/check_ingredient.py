import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests
import tqdm

from datasets import load_dataset

LOGGING_LEVEL = logging.INFO
logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
handler.setFormatter(formatter)
handler.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)


RESPONSE_SAVE_DIR = Path(
    "/home/raphael/Projects/openfoodfacts-ai/ingredient_extraction/classification-responses"
)
RESPONSE_SAVE_DIR.mkdir(exist_ok=True)

http_session = requests.Session()

OPENAI_API_KEY = os.environ["OPENAPI_KEY"]
PROMPT = """Check that the following food ingredient list detected using a ML model is valid. It should:
- be an ingredient list of a food product
- start with the first ingredient and end with the last ingredient. It should not include prefix or suffix information (ex: "Ingredients:").
- it should not include allergy, label or origin information. It's however okay if the allergy, label or origin information is part of the ingredient, listed after the ingredient name, ex: "milk (lactose)".
- if the ingredient list contains a single element (ex: "milk"), it should be extracted only if there is an "ingredient" prefix (or the translation of "ingredient" in another language)
Ingredients list can be in any language, not only in English.

The output format must be a JSON document containing 2 fields: `valid` and `reason`:
- `valid`: a bool indicating whether the ingredient list is valid with respect with the rules stated above
- `reason`: if valid is false, gives the reason (as a string) of why the ingredient list not valid. It must be null if valid is true.
Don't include any other text than the expected JSON document.

Input: %s"""


class ContextLengthExceededException(Exception):
    pass


def send_request(text: str) -> dict:
    user_message = PROMPT % text
    payload = {
        "model": "gpt-4-turbo-preview",
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
            logger.info("Error: %s, %d", r.text, r.status_code)
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


def run_check_ingredient(identifier: str, text: str) -> Optional[dict]:
    try:
        response = send_request(text)
    except ContextLengthExceededException:
        logger.info(f"Context length exception for {identifier}")
        return None

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
    return parsed_json


def check_ingredient(item: dict) -> None:
    id_ = item["meta"]["id"]
    if response_exists(id_):
        return

    full_response = {"id": id_, "valid": True, "items": []}
    for start_offset, end_offset in item["offsets"]:
        text = item["text"][start_offset:end_offset]
        logger.debug("Ingredient text: %s", text)
        partial_id = f"{id_}_{start_offset}_{end_offset}"
        response = run_check_ingredient(partial_id, text)
        if not response["valid"]:
            full_response["valid"] = False
        full_response["items"].append(response)

    save_response(id_, full_response)


if __name__ == "__main__":
    DATASET_URLS = {
        "train": "https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-v6_train.jsonl.gz",
        "test": "https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-alpha-v6_test.jsonl.gz",
    }

    base_ds = load_dataset("json", data_files=DATASET_URLS)

    for split in ("train", "test"):
        for sample in tqdm.tqdm(base_ds[split], desc="sample"):
            check_ingredient(sample)
