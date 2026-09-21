import dataclasses
import json
import typing
from typing import Any

from llm_evals.agent import EvaluationAgent
from llm_evals.default_task import (
    _load_cached_response,
    _save_cached_response,
    get_query_cache_path,
    run_on_sample,
)
from llm_evals.tasks.prices.receipt_anonymization.schemas import PersonalInfoList


async def task_func(inputs: dict[str, Any]) -> str:
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
        output = PersonalInfoList.model_validate_json(resp)
        return resp

    # If not, call the function and store the result in cache
    result = await run_on_sample(
        image_urls=image_urls, instructions=evaluation_agent.instructions
    )
    usage = result.usage

    output = typing.cast(PersonalInfoList, result.output)
    json_response = output.model_dump_json()
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
