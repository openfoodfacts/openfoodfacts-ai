import json
from typing import Any

from pydantic_ai import ImageUrl

from llm_evals.agent import EvaluationAgent
from llm_evals.cache import cache_llm_request_async
from llm_evals.types import OutputMode

DEFAULT_INSTRUCTIONS = "Predict the categories of this product."


@cache_llm_request_async
async def run(
    *,
    image_urls: list[str],
    instructions: str,
    model: str,
    task_name: str,
    json_schema: str,
    output_mode: OutputMode,
) -> str:
    evaluation_agent = EvaluationAgent.get()
    agent = evaluation_agent.agent
    content = [instructions] + [ImageUrl(url) for url in image_urls]
    resp = await agent.run(content)
    return resp.output.model_dump_json()


async def task(inputs: dict[str, Any]) -> Any:
    evaluation_agent = EvaluationAgent.get()
    return await run(
        image_urls=inputs["image_urls"],
        instructions=evaluation_agent.instructions,
        model=evaluation_agent.model,
        task_name=evaluation_agent.task_name,
        json_schema=json.dumps(evaluation_agent.output_type.model_json_schema()),
        output_mode=evaluation_agent.output_mode,
    )
