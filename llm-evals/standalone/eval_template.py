# /// script
# dependencies = [
#     "deepdiff>=8.6.1",
#     "orjson>=3.11.4",
#     "pillow>=12.2.0",
#     "pydantic-ai>=1.104.0",
#     "pydantic-evals>=1.104.0",
#     "typer",
#     "r-llm-evals",
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

import typer
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ThinkingEffort
from pydantic_evals import Case, Dataset, increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from r_llm_evals import get_model_provider
from r_llm_evals.cache import ModelOutputCache

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
    chat_model = OpenAIChatModel(model, provider=get_model_provider())
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
