from pathlib import Path
from typing import Any

import typer

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
    agent = task_config["agent"]
    func_argument: dict[str, Any] = (
        {"image_urls": image_urls}
        if task_config.get("multiple_images", False)
        else {"image_url": image_urls[0]}
    )

    with agent.override(model=model):
        result = asyncio.run(task_func(func_argument))
        print(result)


@app.command()
def evaluate(
    task: TaskType,
    model: str = DEFAULT_MODEL,
    output_path: Path | None = None,
    include_tags: list[str] | None = typer.Option(
        default=None, help="List of tags to include in the evaluation report."
    ),
    only_errors: bool = typer.Option(
        default=False, help="Whether to include only error cases in the report."
    ),
    include_output: bool = True,
    include_expected_output: bool = True,
    include_reasons: bool = True,
    max_concurrency: int | None = typer.Option(
        default=None, help="Maximum number of concurrent requests to the LLM API."
    ),
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
        only_errors=only_errors,
        max_concurrency=max_concurrency,
    )


if __name__ == "__main__":
    app()
