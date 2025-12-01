import json
from typing import Any

import requests
from pydantic_ai import Agent, BinaryContent, ImageUrl

from llm_evals.cache import cache_llm_request_async
from llm_evals.utils import get_model_name

from .schemas import Label

DEFAUT_MODEL = "google-vertex:gemini-2.5-flash-lite"

DEFAULT_INSTRUCTIONS = (
    "Here is one picture containing a price label, extract information "
    "from it. If you cannot decode an attribute, set it to an empty string."
)
agent = Agent(model=DEFAUT_MODEL, output_type=Label)


@cache_llm_request_async
async def run(
    *,
    image_url: str,
    instructions: str,
    model: str,
    task_name: str,
    json_schema: str,
) -> str:
    image_obj: BinaryContent | ImageUrl

    if image_url.startswith(
        "https://robotoff.openfoodfacts.org/api/v1/images/crop"
    ) or image_url.startswith("https://robotoff.openfoodfacts.net/api/v1/images/crop"):
        # Robotoff crop images are not accessible publicly, so we replace them
        # with the original image
        image_obj = BinaryContent(
            requests.get(image_url).content, media_type="image/jpeg"
        )
    else:
        image_obj = ImageUrl(image_url)

    resp = await agent.run(
        [
            instructions,
            image_obj,
        ]
    )
    return resp.output.model_dump_json()


async def task(inputs: dict[str, Any]) -> Any:
    model_name = get_model_name(agent)
    instructions = DEFAULT_INSTRUCTIONS
    return await run(
        image_url=inputs["image_url"],
        instructions=instructions,
        model=model_name,
        task_name="price_price_tag_extraction",
        json_schema=json.dumps(Label.model_json_schema()),
    )
