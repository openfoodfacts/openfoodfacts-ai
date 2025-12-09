import json
from typing import Any

from pydantic_ai import ImageUrl

from llm_evals.agent import EvaluationAgent
from llm_evals.cache import cache_llm_request_async
from llm_evals.types import OutputMode


@cache_llm_request_async
async def run_on_sample(
    *,
    image_urls: list[str],
    instructions: str,
    model: str,
    task_name: str,
    json_schema: str,
    output_mode: OutputMode,
    thinking_config: str | None = None,
) -> str:
    evaluation_agent = EvaluationAgent.get()
    agent = evaluation_agent.agent
    image_content = [ImageUrl(url) for url in image_urls]
    resp = await agent.run(
        [instructions] + image_content,
    )
    return resp.output.model_dump_json()


async def default_task_func(inputs: dict[str, Any]) -> Any:
    evaluation_agent = EvaluationAgent.get()
    return await run_on_sample(
        image_urls=inputs["image_urls"],
        instructions=evaluation_agent.instructions,
        model=evaluation_agent.model,
        task_name=evaluation_agent.task_name,
        json_schema=json.dumps(evaluation_agent.output_type.model_json_schema()),
        output_mode=evaluation_agent.output_mode,
        thinking_config=evaluation_agent.thinking_config,
    )
