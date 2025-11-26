from typing import Any, Callable

from pydantic_ai import Agent
from pydantic_evals import Dataset

from llm_evals.tasks.food.product_info_extraction.config import (
    CONFIG as food_product_info_extraction_config,
)
from llm_evals.tasks.prices.price_tag_extraction.config import (
    CONFIG as prices_price_tag_extraction_config,
)
from llm_evals.types import TaskType

TASK_CONFIG_MAPPING: dict[TaskType, dict] = {
    "food:product_info_extraction": food_product_info_extraction_config,
    "prices:price_tag_extraction": prices_price_tag_extraction_config,
}


def evaluate_task_on_dataset(
    model: str,
    agent: Agent[None, Any],
    dataset: Dataset,
    task_func: Callable,
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
) -> None:
    task_name = f"{task_func.__name__}_{model}"
    with agent.override(model=model):
        report = dataset.evaluate_sync(
            name=task_name,
            task=task_func,
        )
    report.print(
        include_output=include_output,
        include_expected_output=include_expected_output,
        include_reasons=include_reasons,
    )


def evaluate_task(model: str, task: TaskType, **kwargs) -> None:
    """Evaluate a specific task with the given model.
    The Agent (model, prompt, output schema), Dataset and task function are
    retrieved from the task configuration.

    Args:
        model (str): The model to evaluate.
        task (TaskType): The task to evaluate.
        **kwargs: Additional keyword arguments to pass to the
            EvaluationReport.print function.
    """
    task_config = TASK_CONFIG_MAPPING[task]
    evaluate_task_on_dataset(
        model=model,
        agent=task_config["agent"],
        dataset=task_config["dataset"],
        task_func=task_config["task"],
        **kwargs,
    )
