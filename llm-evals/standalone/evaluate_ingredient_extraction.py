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
"""Evaluate ingredient extraction from Open Food Facts images.

Define a .env file containing the necessary credentials (OPENAI_BASE_URL and OPENAI_API_KEY)
Then run the evaluation script:

```bash
uv run --env-file=.env evaluate_ingredient_extraction.py qwen3.5-397b-a17b
```
"""

import typing
from dataclasses import dataclass

import typer
from llm_eval import ModelOutputCache, get_diff, normalize
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ImageUrl, PromptedOutput
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ThinkingEffort
from pydantic_evals import Dataset, increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)


class IngredientList(BaseModel):
    language: str = Field(
        description="Language (ISO 639-1 code) of the ingredient list"
    )
    ingredients: str = Field(
        description="Full ingredient list text. The ingredient text should not include "
        "any prefix such as 'Ingredients:' or 'Ingredients list:'. It should only "
        "include the actual list of ingredients. Allergen mentions should be included "
        "in the ingredient list only if they are present just after the ingredient list, "
        "and not somewhere else on the product. Do *NOT* reconstruct missing parts of "
        "the ingredient list if it is truncated or occluded on the image, just return the "
        "visible part of the ingredient list. Do *NOT* modify the ingredient list in any "
        "way.",
    )
    truncated: bool = Field(
        description="Whether the ingredient list is truncated or occluded on the image",
    )


class Output(BaseModel):
    ingredient_lists: list[IngredientList] = Field(
        description="List of ingredient lists in different languages. If no ingredient list is present, return an empty list.",
    )


OUTPUT_TYPE = Output
TASK_NAME = "ingredient_extraction"


ExpectedOutput = Output


class Inputs(BaseModel):
    image_url: str


class MetaData(BaseModel):
    difficulty: str | None = None
    tags: list[str] | None = None


def get_instructions(inputs: Inputs) -> str | list:
    """Return the agent instructions for the given inputs.
    The instructions must include the input."""
    instructions = (
        "Extract all ingredient lists from the image. If no packaging of a food "
        "product is present on the image, return an empty list."
    )
    return [instructions, ImageUrl(url=inputs.image_url)]


@dataclass
class CustomEvaluator(Evaluator[Inputs, ExpectedOutput, MetaData]):
    def evaluate(
        self, ctx: EvaluatorContext[Inputs, ExpectedOutput, MetaData]
    ) -> bool | EvaluationReason:
        if not ctx.expected_output:
            raise RuntimeError("expected_output should not be null")

        output = ctx.output
        expected_output = ctx.expected_output
        if expected_output != output:
            if len(output.ingredient_lists) != len(expected_output.ingredient_lists):
                return EvaluationReason(
                    value=False,
                    reason=f"output has {len(output.ingredient_lists)} ingredient lists, expected {len(expected_output.ingredient_lists)}",
                )
            for i, (output_list, expected_list) in enumerate(
                zip(
                    output.ingredient_lists,
                    expected_output.ingredient_lists,
                    strict=True,
                )
            ):
                if output_list != expected_list:
                    if output_list.language != expected_list.language:
                        return EvaluationReason(
                            value=False,
                            reason=f"language mismatch: output={output_list.language}, expected={expected_list.language}",
                        )
                    if output_list.truncated != expected_list.truncated:
                        return EvaluationReason(
                            value=False,
                            reason=f"truncated mismatch for lang {output_list.language}: output={output_list.truncated}, expected={expected_list.truncated}",
                        )

                    if output_list.truncated:
                        continue
                    normalized_output = normalize(output_list.ingredients)
                    normalized_expected = normalize(expected_list.ingredients)
                    diff = get_diff(normalized_output, normalized_expected)
                    if diff:
                        return EvaluationReason(
                            value=False,
                            reason=f"ingredients mismatch\ndiff:\n{diff}\noutput:\n{output_list.ingredients}\nexpected:\n{expected_list.ingredients}",
                        )
        return True


dataset = Dataset[Inputs, ExpectedOutput, MetaData].from_file(
    "ingredient_extraction.yaml", custom_evaluator_types=[CustomEvaluator]
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
        increment_eval_metric("input_tokens", response.usage.input_tokens)
        increment_eval_metric("output_tokens", response.usage.output_tokens)
        result = typing.cast(OUTPUT_TYPE, response.output)
        llm_output_cache.save_to_cache(
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
