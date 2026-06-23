# /// script
# dependencies = [
#     "deepdiff>=8.6.1",
#     "orjson>=3.11.4",
#     "pillow>=12.2.0",
#     "pydantic-ai>=1.104.0",
#     "pydantic-evals>=1.104.0",
#     "typer",
# ]
# ///
"""This template can be used to quickly set up an evaluation benchmark for a LLM model
on any task.

Inputs can be image, text, or a combination of both.
The LLM is requested to generate an output that follows a specific JSON schema, defined
by the `Output` class below.

To use this template:

- Define the task name in the `TASK_NAME` variable.
- Define the expected schema in the `Output` class below.
- Define the schema of the expected output in the `ExpectedOutput` class below.
- Update if necessary the metadata schema associated with each case in the `Metadata`
  class below.
- Define the instructions for the LLM (that depends on the inputs) in the
  `get_instructions` function below.
- Evaluate the LLM results using the CustomEvaluator class below.
"""

import typing
from dataclasses import dataclass
from pathlib import Path

import orjson
import typer
from deepdiff import DeepHash
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ThinkingEffort, ThinkingLevel
from pydantic_evals import Case, Dataset, increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)

# Define schema of Output, ExpectedOutput, and Inputs classes below
# Name the task in TASK_NAME


class Output(BaseModel):
    field: str


OUTPUT_TYPE = Output
TASK_NAME = "your_task"


class ExpectedOutput(BaseModel):
    field: str


class Inputs(BaseModel):
    input: str


class MetaData(BaseModel):
    difficulty: str | None = None


def get_instructions(inputs: Inputs) -> str | list:
    """Return the agent instructions for the given inputs.
    The instructions must include the input."""
    return "instructions"


class ModelOutputCache[BaseModelType: BaseModel]:
    def __init__(
        self,
        task_name: str,
        output_type: type[BaseModelType],
        cache_dir: Path | None = None,
    ):
        if cache_dir is None:
            cache_dir = Path("~/.cache/llm_evals").expanduser()
        self.cache_dir = cache_dir
        self.task_name = task_name
        self.output_type = output_type

    def get_query_cache_path(
        self,
        *,
        inputs: Inputs,
        model: str,
        instructions: str | list,
        output_mode: str,
        thinking_effort: ThinkingLevel,
    ) -> Path:
        json_schema = self.output_type.model_json_schema()
        model = model.replace("/", "_")
        cache_key = (
            inputs,
            model,
            self.task_name,
            instructions,
            json_schema,
            output_mode,
            thinking_effort,
        )
        cache_sha256 = DeepHash(cache_key)[cache_key]

        # Split the cache sha256 into subdirectories for better file system
        # performance
        cache_sha256_str = str(cache_sha256)
        subdirs = [cache_sha256_str[i : i + 2] for i in range(0, 6, 2)]
        cache_subdir = Path(*subdirs)
        full_cache_dir = self.cache_dir / self.task_name / model / cache_subdir
        return full_cache_dir / f"{cache_sha256}.json"

    def check_cache(
        self,
        *,
        inputs: Inputs,
        model: str,
        instructions: str | list,
        output_mode: str,
        thinking_effort: ThinkingLevel,
    ) -> BaseModelType | None:
        query_cache_path = self.get_query_cache_path(
            inputs=inputs,
            model=model,
            instructions=instructions,
            output_mode=output_mode,
            thinking_effort=thinking_effort,
        )
        if query_cache_path.exists():
            output = orjson.loads(query_cache_path.read_bytes())["output"]
            return self.output_type.model_validate(output)
        return None

    def save_to_cache(
        self,
        *,
        inputs: Inputs,
        model: str,
        instructions: str | list,
        output_mode: str,
        thinking_effort: ThinkingLevel,
        output: BaseModelType,
    ) -> None:
        query_cache_path = self.get_query_cache_path(
            inputs=inputs,
            model=model,
            instructions=instructions,
            output_mode=output_mode,
            thinking_effort=thinking_effort,
        )
        query_cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "inputs": inputs.model_dump(),
            "output": output.model_dump(),
            "model": model,
            "task_name": self.task_name,
            "thinking_effort": thinking_effort,
            "instructions": instructions,
            "output_mode": output_mode,
            "json_schema": self.output_type.model_json_schema(),
        }
        with query_cache_path.open("wb") as f:
            f.write(orjson.dumps(data))


@dataclass
class CustomEvaluator(Evaluator[Inputs, ExpectedOutput, MetaData]):
    def evaluate(
        self, ctx: EvaluatorContext[Inputs, ExpectedOutput, MetaData]
    ) -> bool | EvaluationReason:
        if not ctx.expected_output:
            raise RuntimeError("expected_output should not be null")

        output = typing.cast(OUTPUT_TYPE, ctx.output)
        return output.field == ctx.expected_output.field


dataset = Dataset[Inputs, ExpectedOutput, MetaData](
    name=TASK_NAME,
    cases=[
        Case(
            name="case_name",
            inputs=Inputs(input="input"),
            expected_output=ExpectedOutput(field="expected_output"),
        ),
    ],
    evaluators=[
        CustomEvaluator(),
    ],
)


def evaluate(
    model: str,
    thinking_effort: ThinkingEffort = "minimal",
    include_output: bool = False,
    include_reasons: bool = True,
):
    chat_model = OpenAIChatModel(model, provider=OpenAIProvider())
    output_type = PromptedOutput(OUTPUT_TYPE)
    agent = Agent(
        chat_model,
        output_type=output_type,
        capabilities=[Thinking(effort=thinking_effort)],
    )
    llm_output_cache = ModelOutputCache(task_name=TASK_NAME, output_type=OUTPUT_TYPE)

    async def run_task(inputs: Inputs) -> OUTPUT_TYPE:
        instructions = get_instructions(inputs)

        if cached := llm_output_cache.check_cache(
            inputs=inputs,
            model=model,
            instructions=instructions,
            output_mode="prompted",
            thinking_effort=thinking_effort,
        ):
            set_eval_attribute("cache_hit", True)
            return cached

        set_eval_attribute("cache_hit", False)
        increment_eval_metric("api_calls", 1)
        response = await agent.run(user_prompt=instructions)
        increment_eval_metric("tokens", response.usage.total_tokens)
        result = typing.cast(OUTPUT_TYPE, response.output)
        llm_output_cache.save_to_cache(
            inputs=inputs,
            model=model,
            instructions=instructions,
            output_mode="prompted",
            thinking_effort=thinking_effort,
            output=result,
        )
        return result

    # Run the evaluation
    report = dataset.evaluate_sync(run_task)
    # Print the results
    report.print(include_output=include_output, include_reasons=include_reasons)


if __name__ == "__main__":
    typer.run(evaluate)
