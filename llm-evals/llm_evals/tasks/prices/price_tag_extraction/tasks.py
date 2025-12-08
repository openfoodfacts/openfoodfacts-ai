import json
from typing import Any

import aiohttp
from pydantic_ai import BinaryContent, ImageUrl

from llm_evals.agent import EvaluationAgent
from llm_evals.cache import cache_llm_request_async

DEFAULT_INSTRUCTIONS = (
    "Here is one picture containing a price label, extract information "
    "from it. If you cannot decode an attribute, set it to an empty string."
)


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
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                content_bytes = await response.read()
                image_obj = BinaryContent(content_bytes, media_type="image/jpeg")
    else:
        image_obj = ImageUrl(image_url)

    evaluation_agent = EvaluationAgent.get()
    agent = evaluation_agent.agent
    resp = await agent.run(
        [
            instructions,
            image_obj,
        ]
    )
    return resp.output.model_dump_json()


async def task(inputs: dict[str, Any]) -> Any:
    evaluation_agent = EvaluationAgent.get()
    return await run(
        image_url=inputs["image_url"],
        instructions=evaluation_agent.instructions,
        model=evaluation_agent.model,
        task_name="price_price_tag_extraction",
        json_schema=json.dumps(evaluation_agent.output_type.model_json_schema()),
    )
