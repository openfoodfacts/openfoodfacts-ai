import json
from pathlib import Path
from typing import Annotated, Any

import typer

from llm_evals.agent import EvaluationAgent
from llm_evals.types import TaskType

DEFAULT_MODEL = "google-vertex:gemini-2.5-flash-lite"

app = typer.Typer()


@app.command()
def run_task(
    image_urls: list[str] = typer.Argument(
        default=..., help="The image URL(s) to run the task on."
    ),
    model: str = typer.Option(
        default=DEFAULT_MODEL, help="The model to use for the task."
    ),
    task: TaskType = typer.Option(
        default="food:product_info_extraction", help="The task to run."
    ),
):
    """Run a specific task with the given model on a single image URL.

    The Agent (model, prompt, output schema) and task function are
    retrieved from the task configuration.

    Example:
    python main.py run-task
    "https://images.openfoodfacts.org/images/products/932/721/500/0085/1.jpg"
    --model "google-vertex:gemini-2.5-flash-lite"
    --task "food:product_info_extraction"
    """
    import asyncio

    from llm_evals.evaluate import TASK_CONFIG_MAPPING

    task_config = TASK_CONFIG_MAPPING[task]
    task_func = task_config["task"]
    instructions = task_config["instructions"]
    agent = task_config["agent"]
    func_argument: dict[str, Any] = (
        {"image_urls": image_urls}
        if task_config.get("multiple_images", False)
        else {"image_url": image_urls[0]}
    )

    EvaluationAgent.create(model_name=model, instructions=instructions)
    with agent.override(model=model):
        result = asyncio.run(task_func(func_argument))

        try:
            json.loads(result)
        except json.JSONDecodeError:
            typer.echo(result)
        else:
            typer.echo(json.dumps(json.loads(result), indent=2, ensure_ascii=False))


@app.command()
def evaluate(
    task: Annotated[TaskType, typer.Argument(..., help="The task to evaluate.")],
    model: Annotated[
        str, typer.Option(..., help="The model to evaluate.")
    ] = DEFAULT_MODEL,
    output_path: Annotated[
        Path | None,
        typer.Option(..., help="Path to save the evaluation report as a JSON file."),
    ] = None,
    include_tags: Annotated[
        list[str] | None,
        typer.Option(..., help="List of tags to include in the evaluation report."),
    ] = None,
    only_errors: Annotated[
        bool,
        typer.Option(
            ..., help="Whether to display only error cases in the CLI report."
        ),
    ] = False,
    include_output: Annotated[
        bool,
        typer.Option(
            ..., help="Whether to display the model output in the CLI report."
        ),
    ] = False,
    include_durations: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to display the durations for each case in the CLI report.",
        ),
    ] = False,
    include_input: Annotated[
        bool,
        typer.Option(..., help="Whether to display the input in the CLI report."),
    ] = False,
    include_expected_output: Annotated[
        bool,
        typer.Option(
            ..., help="Whether to display the expected output in the CLI report."
        ),
    ] = False,
    include_reasons: Annotated[
        bool,
        typer.Option(..., help="Whether to display the reasons for each assertion."),
    ] = True,
    max_concurrency: Annotated[
        int | None,
        typer.Option(..., help="Maximum number of concurrent requests to the LLM API."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option(..., help="Limit the number of samples to evaluate.")
    ] = None,
):
    """Evaluate a specific task with the given model.
    The Agent (model, prompt, output schema), Dataset and task function are
    retrieved from the task configuration.

    Example:
    python main.py evaluate
    --model "google-vertex:gemini-2.5-flash-lite"
    --task "food:product_info_extraction"
    """
    from llm_evals.evaluate import evaluate_task

    evaluate_task(
        model=model,
        task=task,
        include_output=include_output,
        include_expected_output=include_expected_output,
        include_reasons=include_reasons,
        output_path=output_path,
        include_tags=include_tags,
        include_input=include_input,
        include_durations=include_durations,
        only_errors=only_errors,
        max_concurrency=max_concurrency,
        limit=limit,
    )


if __name__ == "__main__":
    app()
