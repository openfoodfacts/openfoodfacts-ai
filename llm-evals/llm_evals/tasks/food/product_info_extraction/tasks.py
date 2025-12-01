from typing import Any

from pydantic_ai import Agent, ImageUrl

from llm_evals.cache import cache_llm_request_async
from llm_evals.utils import get_model_name

from .schemas import ProductInfoExtractionResponseModel

DEFAUT_MODEL = "google-vertex:gemini-2.5-flash-lite"

DEFAULT_INSTRUCTIONS = (
    "Extract all relevant information from this product packaging photo."
)
agent = Agent(model=DEFAUT_MODEL, output_type=ProductInfoExtractionResponseModel)


@cache_llm_request_async
async def run(image_url: str, instruction: str, model: str, task_name: str) -> str:
    resp = await agent.run([instruction, ImageUrl(image_url)])
    return resp.output.model_dump_json()


async def task(inputs: dict[str, Any]) -> Any:
    model_name = get_model_name(agent)
    return await run(
        image_url=inputs["image_url"],
        instruction=DEFAULT_INSTRUCTIONS,
        model=model_name,
        task_name="food_product_info_extraction",
    )
