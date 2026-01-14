from pathlib import Path
from typing import Annotated

import typer

from llm_evals.types import OutputMode, TaskType

DEFAULT_MODEL = "google-vertex:gemini-2.5-flash-lite"


app = typer.Typer()


@app.command()
def from_api(
    task: Annotated[TaskType, typer.Option(..., help="The task to evaluate.")],
    model: Annotated[
        str, typer.Option(..., help="The model to evaluate.")
    ] = DEFAULT_MODEL,
    output_mode: Annotated[
        OutputMode, typer.Option(..., help="The output mode of the agent.")
    ] = "tool",
    thinking_config: Annotated[
        str | None,
        typer.Option(
            ..., help="Optional configuration for the agent's thinking process."
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(..., help="Path to save the evaluation report as a JSON file."),
    ] = None,
    filter: Annotated[
        str | None,
        typer.Option(..., help="Query to filter the cases, using `dictquery` syntax."),
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
        output_mode=output_mode,
        thinking_config=thinking_config,
        include_output=include_output,
        include_expected_output=include_expected_output,
        include_reasons=include_reasons,
        output_path=output_path,
        filter=filter,
        include_input=include_input,
        include_durations=include_durations,
        only_errors=only_errors,
        max_concurrency=max_concurrency,
        limit=limit,
    )


@app.command()
def from_prediction_file(
    task: Annotated[TaskType, typer.Option(..., help="The task to evaluate.")],
    prediction_path: Annotated[
        Path | None,
        typer.Option(
            ...,
            help="Path to a JSONL file containing the model predictions. "
            "The file must contain one item per line, corresponding to a model "
            "prediction on a sample. Each line must have the following fields:\n"
            "- `image_id` (str): the ID of the image, as set in the sample found in the "
            "original HF dataset that was used for training. This value is compared "
            "to the `name` field in each dataset case, to match the case with the "
            "model prediction.\n"
            "- `output`: the model generated output (str)",
        ),
    ] = None,
    hf_repo_id: Annotated[
        str | None,
        typer.Option(
            ...,
            help="ID of a Hugging Face model repository, containing a "
            "`validation_output.jsonl` JSONL file at the root of the repo. "
            "The file will be downloaded and be used instead of the file "
            "provided by `--prediction-path`. "
            "See --prediction-path help message for a description of the expected "
            "file structure. "
            "Only one of `--hf-repo-id` or `--prediction-path` can be set.",
        ),
    ] = None,
    validation_file_name: Annotated[
        str,
        typer.Option(
            ...,
            help="Name of the validation file inside the HF repo. Default is "
            "`validation_output.jsonl`. This option is only used when "
            "`--hf-repo-id` is set.",
        ),
    ] = "validation_output.jsonl",
    output_path: Annotated[
        Path | None,
        typer.Option(..., help="Path to save the evaluation report as a JSON file."),
    ] = None,
    filter: Annotated[
        str | None,
        typer.Option(..., help="Query to filter the cases, using `dictquery` syntax."),
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
    limit: Annotated[
        int | None, typer.Option(..., help="Limit the number of samples to evaluate.")
    ] = None,
):
    """Evaluate a specific task on a model using predictions stored in a JSONL
    file."""
    import typing

    from huggingface_hub import hf_hub_download

    from llm_evals.evaluate import evaluate_task_from_file

    if prediction_path is None and hf_repo_id is None:
        typer.echo(
            "One of `--prediction-path` or `--hf-repo-id` must be provided", err=True
        )
        raise typer.Exit(1)

    if prediction_path is not None and hf_repo_id is not None:
        typer.echo(
            "Only one of `--prediction-path` and `--hf-repo-id` must be provided",
            err=True,
        )
        raise typer.Exit(1)

    if hf_repo_id is not None:
        prediction_path_str = hf_hub_download(
            repo_id=hf_repo_id, filename=validation_file_name
        )

        if prediction_path_str is None:
            typer.echo(
                f"File `validation_output.jsonl` not found in repo {hf_repo_id}. Make sure that the file exists.",
                err=True,
            )
            raise typer.Exit(1)

        prediction_path = Path(prediction_path_str)

    evaluate_task_from_file(
        prediction_path=typing.cast(Path, prediction_path),
        task=task,
        include_output=include_output,
        include_expected_output=include_expected_output,
        include_reasons=include_reasons,
        output_path=output_path,
        filter=filter,
        include_input=include_input,
        include_durations=include_durations,
        only_errors=only_errors,
        limit=limit,
    )
