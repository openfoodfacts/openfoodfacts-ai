import dataclasses
import json
from typing import Any

from pydantic_ai import AgentRunResult, ImageUrl

from llm_evals.agent import EvaluationAgent
from llm_evals.cache import (
    _load_cached_response,
    _save_cached_response,
    disk_cache,
    get_query_cache_path,
)


async def run_on_sample(*, image_urls: list[str], instructions: str) -> AgentRunResult:
    evaluation_agent = EvaluationAgent.get()
    agent = evaluation_agent.agent
    image_content = [ImageUrl(url) for url in image_urls]
    return await agent.run(
        [instructions] + image_content,
    )


async def default_task_func(inputs: dict[str, Any]) -> str:
    evaluation_agent = EvaluationAgent.get()
    image_urls = inputs["image_urls"]
    json_schema = json.dumps(evaluation_agent.output_type.model_json_schema())
    query_cache_path = get_query_cache_path(
        image_urls=image_urls,
        model=evaluation_agent.model,
        task_name=evaluation_agent.task_name,
        instructions=evaluation_agent.instructions,
        json_schema=json_schema,
        output_mode=evaluation_agent.output_mode,
        thinking_config=evaluation_agent.thinking_config,
    )

    if (resp := _load_cached_response(query_cache_path)) is not None:
        return resp

    # If not, call the function and store the result in cache
    result = await run_on_sample(
        image_urls=image_urls, instructions=evaluation_agent.instructions
    )
    usage = result.usage()

    json_response = result.output.model_dump_json()
    data = {
        "image_urls": image_urls,
        "output": json_response,
        "model": evaluation_agent.model,
        "task_name": evaluation_agent.task_name,
        "thinking_config": evaluation_agent.thinking_config,
        "instructions": evaluation_agent.instructions,
        "output_mode": evaluation_agent.output_mode,
        "json_schema": json.loads(json_schema),
        "usage": dataclasses.asdict(usage) if usage else None,
    }
    _save_cached_response(query_cache_path, data)
    return json_response


async def default_task_func_from_file(inputs: dict[str, Any]) -> str:
    """Task function that retrieves the model output from the disk cache
    and returns it.

    Args:
        inputs: A dictionary containing the input data for the task function.
            We expect it to contain the key `image_id`.
    Returns:
        The model output as a string.
    """
    image_id = inputs["image_id"]
    output: str | None = disk_cache.get(image_id)
    if output is None:
        raise ValueError(
            f"missing key from cache: '{image_id}'. "
            "Ensure that all samples in the dataset can be found in the JSONL prediction file."
        )
    return output
