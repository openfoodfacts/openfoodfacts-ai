import dataclasses
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import orjson
import typer
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport

from llm_evals.agent import EvaluationAgent
from llm_evals.tasks.food.product_categorization.config import (
    CONFIG as food_product_categorization_config,
)
from llm_evals.tasks.food.product_info_extraction.config import (
    CONFIG as food_product_info_extraction_config,
)
from llm_evals.tasks.prices.price_tag_extraction.config import (
    CONFIG as prices_price_tag_extraction_config,
)
from llm_evals.types import TaskConfig, TaskType

TASK_CONFIG_MAPPING: dict[TaskType, TaskConfig] = {
    "food:product_info_extraction": food_product_info_extraction_config,
    "food:product_categorization": food_product_categorization_config,
    "prices:price_tag_extraction": prices_price_tag_extraction_config,
}


def default_serializer(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError


def compute_tag_count(
    report: EvaluationReport,
) -> Counter:
    counter: Counter[str] = Counter()
    for case in report.cases:
        tags = case.metadata.get("tags", []) if case.metadata else []
        counter.update(tags)
    return counter


def compute_assertion_accuracy(
    report: EvaluationReport,
) -> dict[str, tuple[int, int, float]]:
    accuracy_by_assertion: dict[str, tuple[int, int, float]] = {}
    for case in report.cases:
        for assertion_name, assertion in case.assertions.items():
            accuracy_by_assertion.setdefault(assertion_name, (0, 0, 0.0))
            current_correct_count, current_total_count, _ = accuracy_by_assertion[
                assertion_name
            ]
            current_total_count += 1
            if assertion.value is True:
                current_correct_count += 1

            accuracy_by_assertion[assertion_name] = (
                current_correct_count,
                current_total_count,
                0,
            )

    for assertion_name, (
        correct_count,
        total_count,
        __,
    ) in accuracy_by_assertion.items():
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        accuracy_by_assertion[assertion_name] = (correct_count, total_count, accuracy)

    return accuracy_by_assertion


def evaluate_task_on_dataset(
    model: str,
    task_config: TaskConfig,
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
    include_input: bool = False,
    include_durations: bool = False,
    include_tags: list[str] | None = None,
    output_path: Path | None = None,
    only_errors: bool = False,
    max_concurrency: int | None = None,
    limit: int | None = None,
) -> None:
    """Evaluate a specific task on a given dataset with the given model.

    Args:
        model (str): The model to evaluate. Overrides the model in the agent.
        task_config (TaskConfig): The task configuration containing the task
            function and other settings.
        include_output (bool): Whether to include the model output in the
            report.
        include_expected_output (bool): Whether to include the expected output
            in the report.
        include_reasons (bool): Whether to include the reasons for each
            assertion in the report.
        include_input (bool): Whether to include the input in the report.
        include_durations (bool): Whether to include the durations for each
            case in the report.
        include_tags (list[str] | None): List of tags to filter the dataset
            cases. If None, all cases are included.
        output_path (Path | None): Path to save the evaluation report as a
            JSON file. If None, the report is not saved.
        only_errors (bool): Whether to include only error cases in the report.
        max_concurrency (int | None): Maximum number of concurrent requests to
            the LLM API. If None, no limit is set.
        limit (int | None): Limit the number of samples to evaluate. If None,
            all samples are evaluated.
    """
    dataset = task_config["dataset"]
    task_func: Callable[[dict[str, Any]], Any] = task_config["task"]
    if include_tags is not None:
        dataset.cases = [
            case
            for case in dataset.cases
            if case.metadata
            and any(tag in case.metadata.get("tags", []) for tag in include_tags)
        ]

    if limit is not None:
        dataset.cases = dataset.cases[:limit]

    task_name = f"{task_func.__name__}_{model}"
    instructions = task_config["instructions"]
    task_output_type = task_config["output_type"]
    EvaluationAgent.set(
        model=model, instructions=instructions, output_type=task_output_type
    )
    report = dataset.evaluate_sync(
        name=task_name,
        task=task_func,
        max_concurrency=max_concurrency,
    )

    if only_errors:
        report.cases = [
            case
            for case in report.cases
            if not all(
                [assertion.value is True for assertion in case.assertions.values()]
            )
        ]

    report.print(
        include_output=include_output,
        include_expected_output=include_expected_output,
        include_reasons=include_reasons,
        include_durations=include_durations,
        include_input=include_input,
    )
    typer.echo("---" * 15)
    typer.echo(f"Number of cases: {len(report.cases)}")
    typer.echo("---" * 15)
    typer.echo("Detailed scores:")
    accuracy_by_assertion = compute_assertion_accuracy(report)
    for assertion_name, (
        correct_count,
        total_count,
        accuracy,
    ) in accuracy_by_assertion.items():
        typer.echo(
            f"  {assertion_name}: {correct_count}/{total_count} ({accuracy:.2%} accuracy)"
        )

    typer.echo("---" * 15)
    tag_counts = compute_tag_count(report)
    typer.echo("Tag counts in evaluated cases:")
    for tag, count in sorted(
        tag_counts.items(), key=lambda x: (x[0].split(":", 1)[0], -x[1])
    ):
        typer.echo(f"  {tag}: {count}")

    if output_path:
        output_path.write_bytes(
            orjson.dumps(
                dataclasses.asdict(report),
                default=default_serializer,
                option=orjson.OPT_INDENT_2,
            )
        )

    report.cases


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
        task_config=task_config,
        **kwargs,
    )
