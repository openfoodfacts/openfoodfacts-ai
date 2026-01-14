import copy
import dataclasses
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import dictquery
import orjson
import typer
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.reporting import EvaluationReport

from llm_evals.agent import EvaluationAgent
from llm_evals.cache import disk_cache, populate_disk_cache
from llm_evals.default_task import default_task_func, default_task_func_from_file
from llm_evals.tasks.food.product_categorization.config import (
    CONFIG as food_product_categorization_config,
)
from llm_evals.tasks.food.product_info_extraction.config import (
    CONFIG as food_product_info_extraction_config,
)
from llm_evals.tasks.prices.price_tag_extraction.config import (
    CONFIG as prices_price_tag_extraction_config,
)
from llm_evals.types import OutputMode, TaskConfig, TaskType

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
    """Compute the accuracy for each assertion in the evaluation report.
    Args:
        report (EvaluationReport): The evaluation report to compute the
            assertion accuracy from.
    Returns:
        dict[str, tuple[int, int, float]]: A dictionary mapping each assertion
            name to a tuple containing the number of correct assertions,
            the total number of assertions, and the accuracy as a float.
    """
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


def filter_dataset(
    cases: list[Case], filter: str | None, limit: int | None = None
) -> list[Case]:
    """Filter the dataset cases based on the given filter query and
    limit.

    Args:
        cases (list[Case]): The list of dataset cases to filter.
        filter (str | None): The filter query to apply to the cases.
        limit (int | None): The maximum number of cases to return.
            If None, all matching cases are returned.
    Returns:
        list[Case]: The filtered list of dataset cases.
    """
    if filter is not None:
        # Hacky way to avoid having to type '`tags.name` = tag', as:
        # - it's too verbose
        # - we need to escape '`' char in bash, making it even less convenient.
        filter = filter.replace("tags ==", "`metadata.tags.name` ==")

    new_cases = []
    added = 0
    for case in cases:
        if filter is not None:
            tags = case.metadata.get("tags", [])
            if tags:
                formatted_case = copy.deepcopy(case)
                # We cannot filter directly a list of string with dictquery
                # library, only a list of dict. So we transform the string into
                # a dict with a single `name` field, containing the tag value.
                formatted_case.metadata["tags"] = [{"name": tag} for tag in tags]
            else:
                formatted_case = case
            if dictquery.match(dataclasses.asdict(formatted_case), filter):
                new_cases.append(case)
                added += 1
        else:
            new_cases.append(copy.deepcopy(case))
            added += 1

        if limit and added >= limit:
            break

    return new_cases


def process_report(
    report: EvaluationReport,
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
    include_input: bool = False,
    include_durations: bool = False,
    output_path: Path | None = None,
    only_errors: bool = False,
) -> None:
    """Process and print the evaluation report.

    Processing only includes filtering error cases if specified, and printing
    the report with the specified options.

    Args:
        report (EvaluationReport): The evaluation report to process.
        include_output (bool): Whether to include the model output in the
            printed report.
        include_expected_output (bool): Whether to include the expected output
            in the printed report.
        include_reasons (bool): Whether to include the reasons for each
            assertion in the printed report.
        include_input (bool): Whether to include the input in the printed
            report.
        include_durations (bool): Whether to include the durations for each
            case in the printed report.
        output_path (Path | None): Path to save the evaluation report as a
            JSON file. If None, the report is not saved.
        only_errors (bool): Whether to include only error cases in the report.
    """
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


def evaluate_task_on_dataset(
    model: str,
    task_config: TaskConfig,
    output_mode: OutputMode = "tool",
    thinking_config: str | None = None,
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
    include_input: bool = False,
    include_durations: bool = False,
    filter: str | None = None,
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
        output_mode (OutputMode): The output mode of the agent, which can be
            "tool", "native", or "prompted". Defaults to "tool".
        thinking_config (str | None): Optional configuration for the
            agent's thinking process.
        include_output (bool): Whether to include the model output in the
            report.
        include_expected_output (bool): Whether to include the expected output
            in the report.
        include_reasons (bool): Whether to include the reasons for each
            assertion in the report.
        include_input (bool): Whether to include the input in the report.
        include_durations (bool): Whether to include the durations for each
            case in the report.
        filter (str | None): Only run cases that match the filter query.
        output_path (Path | None): Path to save the evaluation report as a
            JSON file. If None, the report is not saved.
        only_errors (bool): Whether to include only error cases in the report.
        max_concurrency (int | None): Maximum number of concurrent requests to
            the LLM API. If None, no limit is set.
        limit (int | None): Limit the number of samples to evaluate. If None,
            all samples are evaluated.
    """
    dataset = task_config.dataset
    task_func: Callable[[dict[str, Any]], Any] = task_config.task or default_task_func
    dataset.cases = filter_dataset(cases=dataset.cases, filter=filter, limit=limit)
    task_name = f"{task_func.__name__}_{model}"
    instructions = task_config.instructions
    EvaluationAgent.set(
        model=model,
        instructions=instructions,
        output_type=task_config.output_type,
        output_mode=output_mode,
        task_name=task_config.name,
        thinking_config=thinking_config,
    )
    report = dataset.evaluate_sync(
        name=task_name,
        task=task_func,
        max_concurrency=max_concurrency,
    )
    process_report(
        report,
        include_output=include_output,
        include_expected_output=include_expected_output,
        include_reasons=include_reasons,
        include_input=include_input,
        include_durations=include_durations,
        only_errors=only_errors,
        output_path=output_path,
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
        task_config=task_config,
        **kwargs,
    )


def evaluate_task_on_dataset_from_file(
    prediction_path: Path,
    dataset: Dataset,
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
    include_input: bool = False,
    include_durations: bool = False,
    filter: str | None = None,
    output_path: Path | None = None,
    only_errors: bool = False,
    limit: int | None = None,
) -> None:
    """Evaluate a specific task on a given dataset from a JSONL prediction
    file.

    Args:
        prediction_path (Path): The Path of the JSONL file containing the
            model predictions.
        dataset (Dataset): The dataset containing the samples to evaluate.
            All samples in this dataset must be present in the prediction file.
            To match each sample in the dataset with a sample in the JSONL
            prediction file, we use the `name` field of each dataset case
            (which represents the image ID) and the `image_id` field in the
            prediction file.
        include_output (bool): Whether to include the model output in the
            report.
        include_expected_output (bool): Whether to include the expected output
            in the report.
        include_reasons (bool): Whether to include the reasons for each
            assertion in the report.
        include_input (bool): Whether to include the input in the report.
        include_durations (bool): Whether to include the durations for each
            case in the report.
        filter (str | None): Only run cases that match the filter query.
        output_path (Path | None): Path to save the evaluation report as a
            JSON file. If None, the report is not saved.
        only_errors (bool): Whether to include only error cases in the report.
        limit (int | None): Limit the number of samples to evaluate. If None,
            all samples are evaluated.
    """
    dataset.cases = filter_dataset(cases=dataset.cases, filter=filter, limit=limit)

    for case in dataset.cases:
        # pydantic_evals only pass the case `inputs` to the task function.
        # As we use the image ID to retrieve the model predictions for this
        # sample from all the samples in the validation set, we dynamically
        # change the inputs here
        image_id = case.name
        case.inputs = {"image_id": str(image_id)}
    with disk_cache:
        # We use diskcache to cache each model output with image ID as the
        # cache key.
        populate_disk_cache(disk_cache, prediction_path)
        report = dataset.evaluate_sync(
            task=default_task_func_from_file,
            # no need for concurrency, as there are no network requests
            max_concurrency=1,
        )
        process_report(
            report,
            include_output=include_output,
            include_expected_output=include_expected_output,
            include_reasons=include_reasons,
            include_input=include_input,
            include_durations=include_durations,
            only_errors=only_errors,
            output_path=output_path,
        )


def evaluate_task_from_file(prediction_path: Path, task: TaskType, **kwargs) -> None:
    """Evaluate a specific task from a JSONL file containing predictions.
    The Dataset is retrieved from the task configuration.

    Args:
        prediction_path (Path): The Path of the JSONL file containing the
            model predictions.
        task (TaskType): The task to evaluate.
        **kwargs: Additional keyword arguments to pass to
            `evaluate_task_on_dataset_from_file` function.
    """
    task_config = TASK_CONFIG_MAPPING[task]
    evaluate_task_on_dataset_from_file(
        prediction_path=prediction_path,
        dataset=task_config.dataset,
        **kwargs,
    )
