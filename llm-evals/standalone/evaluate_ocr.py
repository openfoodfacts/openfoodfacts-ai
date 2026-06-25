# /// script
# dependencies = [
#     "deepdiff>=8.6.1",
#     "orjson>=3.11.4",
#     "pillow>=12.2.0",
#     "pydantic-ai>=1.104.0",
#     "pydantic-evals>=1.104.0",
#      "r-llm-evals",
#     "levenshtein",
#     "typer",
# ]
# ///
"""Evaluate OCR models on Open Food Facts images."""

import typing
from dataclasses import dataclass

import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ImageUrl
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ThinkingEffort
from pydantic_evals import Case, Dataset, increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)
from r_llm_evals import get_model_provider
from r_llm_evals.cache import ModelOutputCache
from r_llm_evals.text import get_diff, normalize

TASK_NAME = "off_ocr"


class ExpectedOutput(BaseModel):
    full_exact_match: str | None = None
    full_fuzzy_match: str | None = None
    partial_exact_matches: list[str] | None = Field(None, min_length=1)
    partial_fuzzy_matches: list[str] | None = Field(None, min_length=1)


class Inputs(BaseModel):
    image_url: str


class MetaData(BaseModel):
    difficulty: str | None = None
    tags: list[str] | None = None


def get_instructions(inputs: Inputs) -> list:
    """Return the agent instructions for the given inputs.
    The instructions must include the input."""
    text = """Convert the following product photo to text.
    Return only the extracted text with no explanation text. Do not include delimiters like ```markdown or ```html.

    RULES:
      - You must include all information on the photo.
      - Don't keep line breaks if the words belong to the same paragraph:
        only add line break when changing paragraph.
    """
    return [
        text,
        ImageUrl(url=inputs.image_url),
    ]


@dataclass
class CustomEvaluator(Evaluator[Inputs, ExpectedOutput, MetaData]):
    def evaluate(
        self, ctx: EvaluatorContext[Inputs, ExpectedOutput, MetaData]
    ) -> bool | EvaluationReason:
        expected_output = ctx.expected_output
        if not expected_output:
            raise RuntimeError("expected_output should not be null")

        output = typing.cast(str, ctx.output)
        normalized_output = normalize(output)

        if expected_output.full_exact_match is not None:
            normalized_expected = normalize(expected_output.full_exact_match)
            diff = get_diff(normalized_expected, normalized_output)
            if diff:
                return EvaluationReason(
                    value=False,
                    reason=f"diff:\n{diff}",
                )

        if expected_output.partial_exact_matches:
            for pattern in expected_output.partial_exact_matches:
                normalized_pattern = normalize(pattern)
                if normalized_pattern not in normalized_output:
                    return EvaluationReason(
                        value=False,
                        reason=f"pattern: {repr(normalized_pattern)} not found in output: {repr(normalized_output)}",
                    )

        return True


dataset = Dataset[Inputs, ExpectedOutput, MetaData](
    name=TASK_NAME,
    cases=[
        Case(
            name="8480000590909_ingredients",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/848/000/059/0909/ingredients_es.15.400.jpg"
            ),
            expected_output=ExpectedOutput(
                full_exact_match=(
                    "SALAMI EXTRA CON PROTEÍNAS DE LA LECHE\n"
                    "Ingredientes: Carne de cerdo, tocino, lactosa, sal, "
                    "proteínas de la leche, dextrosa, especias, aromas, "
                    "aromas de humo, estabilizantes (E-450, E-451), "
                    "antioxidante (E-301), conservadores (E-250, E-252), "
                    "colorante (E-120) y extracto de pimentón.\n"
                    "Envasado en atmósfera protectora."
                ),
            ),
            metadata=MetaData(difficulty="medium", tags=["ingredients"]),
        ),
        Case(
            name="3256228416933_ingredients",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/325/622/841/6933/nutrition_fr.35.400.jpg"
            ),
            expected_output=ExpectedOutput(
                full_exact_match=(
                    "Ingrédients\n"
                    "Fécule de pomme de terre, farine de riz, sucre, pépites de "
                    "chocolat noir 12% (pâte de cacao, sucre, beurre de cacao), "
                    "œufs, margarine végétale (beurre de karité, huile de "
                    "tournesol, eau, sel, jus de citron concentré), inuline, "
                    "poudre d'amande, farine de maïs, noisettes hachées grillées, "
                    "huile de tournesol, miel, poudre à lever : carbonates d'ammonium, sel.\n"
                    "Traces éventuelles de lait."
                ),
            ),
            metadata=MetaData(difficulty="medium", tags=["ingredients"]),
        ),
        Case(
            name="5941355000374",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/594/135/500/0374/nutrition_ro.10.full.jpg"
            ),
            expected_output=ExpectedOutput(
                partial_exact_matches=[
                    "grăsime.",
                    "Ingrediente:",
                    "smântână",
                    ", obținută",
                    "lapte de vacă",
                    "roteine din lapte, culturi lactice selecționate.",
                    "Valori nutriționale medii",
                    "pentru 100 g de produs",
                    "Valoare energetică",
                    "550 kJ/131 kcal",
                    "Grăsime, din care:",
                    "- acizi grași saturați",
                    "7,8 g",
                    "Glucide, din care:",
                    "- zaharuri",
                    "2,9 g",
                    "Proteine",
                    "Sare*",
                    "0,1 g",
                    "* echivalent sodiu prezent în mod natural în lapte",
                    "Originea laptelui: România.",
                    "Procesator/Ambalator/ Distribuitor: Albalact S.A.,",
                    "DN1, km 392+600,",
                    "517293, Oiejdea, jud. Alba, Tel.: 0258.816.738",
                    "www.albalact.ro",
                    "consumator@albalact.ro",
                    "ALBALACT",
                ],
            ),
            metadata=MetaData(difficulty="hard", tags=["ingredients"]),
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
    chat_model = OpenAIChatModel(model, provider=get_model_provider(model))
    agent = Agent(
        chat_model,
        output_type=str,
        capabilities=[Thinking(effort=thinking_effort)],
    )
    llm_output_cache = ModelOutputCache(task_name=TASK_NAME, output_type=str)

    async def run_task(inputs: Inputs) -> str:
        instructions = get_instructions(inputs)

        if cached := llm_output_cache.check_cache(
            model=model,
            instructions=instructions,
            thinking_effort=thinking_effort,
            output_mode=None,
        ):
            set_eval_attribute("cache_hit", True)
            return cached

        set_eval_attribute("cache_hit", False)
        increment_eval_metric("api_calls", 1)
        response = await agent.run(user_prompt=instructions)
        increment_eval_metric("tokens", response.usage.total_tokens)
        increment_eval_metric("input_tokens", response.usage.input_tokens)
        increment_eval_metric("output_tokens", response.usage.output_tokens)
        result = response.output
        llm_output_cache.save_to_cache(
            model=model,
            instructions=instructions,
            thinking_effort=thinking_effort,
            output=result,
            output_mode=None,
        )
        return result

    # Run the evaluation
    report = dataset.evaluate_sync(run_task)
    # Print the results
    report.print(include_output=include_output, include_reasons=include_reasons)


if __name__ == "__main__":
    typer.run(evaluate)
