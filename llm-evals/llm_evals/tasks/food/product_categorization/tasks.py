import json
from typing import Any

from pydantic_ai import Agent, ImageUrl

from llm_evals.cache import cache_llm_request_async
from llm_evals.utils import get_model_name

from .schemas import CategoryPredictionResponseModel

DEFAUT_MODEL = "google-vertex:gemini-2.5-flash-lite"

DEFAULT_INSTRUCTIONS = "Predict the categories of this product."
agent = Agent(model=DEFAUT_MODEL, output_type=CategoryPredictionResponseModel)


@cache_llm_request_async
async def run(
    *,
    image_urls: list[str],
    instructions: str,
    model: str,
    task_name: str,
    json_schema: str,
) -> str:
    content = [instructions] + [ImageUrl(url) for url in image_urls]
    resp = await agent.run(content)
    return resp.output.model_dump_json()


async def task(inputs: dict[str, Any]) -> Any:
    model_name = get_model_name(agent)
    instructions = DEFAULT_INSTRUCTIONS
    return await run(
        image_urls=inputs["image_urls"],
        instructions=instructions,
        model=model_name,
        task_name="product_categorization",
        json_schema=json.dumps(CategoryPredictionResponseModel.model_json_schema()),
    )
