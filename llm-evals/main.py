import json
from typing import Annotated

import typer

from llm_evals.agent import EvaluationAgent
from llm_evals.apps import evaluate as evaluate_app
from llm_evals.default_task import default_task_func
from llm_evals.types import OutputMode, TaskType

DEFAULT_MODEL = "google-vertex:gemini-2.5-flash-lite"

app = typer.Typer()
app.add_typer(
    evaluate_app.app,
    name="evaluate",
    help="Evaluate models, using APIs or from a file.",
)


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
    output_mode: Annotated[
        OutputMode, typer.Option(..., help="The output mode of the agent.")
    ] = "tool",
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
    task_func = task_config.task or default_task_func
    instructions = task_config.instructions
    EvaluationAgent.set(
        model=model,
        instructions=instructions,
        output_type=task_config.output_type,
        output_mode=output_mode,
        task_name=task_config.name,
    )
    result = asyncio.run(task_func({"image_urls": image_urls}))

    try:
        json.loads(result)
    except json.JSONDecodeError:
        typer.echo(result)
    else:
        typer.echo(json.dumps(json.loads(result), indent=2, ensure_ascii=False))


@app.command()
def add_sample(
    task: Annotated[
        TaskType,
        typer.Argument(..., help="The dataset task we should add the sample to."),
    ],
    inputs: Annotated[
        str,
        typer.Argument(
            ...,
            help="The input to the sampler fetcher function. "
            "It is usally an ID, but it's project dependent.",
        ),
    ],
):
    """Add a sample to an existing dataset."""
    from llm_evals.evaluate import TASK_CONFIG_MAPPING

    task_config = TASK_CONFIG_MAPPING[task]

    if task_config.add_sample_func is None:
        typer.echo(f"No function to add sample is available for task '{task}'")
        return

    task_config.add_sample_func(inputs)


if __name__ == "__main__":
    app()
