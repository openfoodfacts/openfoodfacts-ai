# /// script
# dependencies = [
#     "deepdiff>=8.6.1",
#     "orjson>=3.11.4",
#     "pillow>=12.2.0",
#     "pydantic-ai>=1.104.0",
#     "pydantic-evals>=1.104.0",
#     "levenshtein",
#     "typer",
# ]
# ///
"""Evaluate OCR models on Open Food Facts images."""

import re
import typing
from dataclasses import dataclass
from difflib import Differ
from pathlib import Path

import orjson
import typer
from deepdiff import DeepHash
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ImageUrl
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


class ModelOutputCache:
    def __init__(
        self,
        task_name: str,
        cache_dir: Path | None = None,
    ):
        if cache_dir is None:
            cache_dir = Path("~/.cache/llm_evals").expanduser()
        self.cache_dir = cache_dir
        self.task_name = task_name

    def get_query_cache_path(
        self,
        *,
        model: str,
        instructions: list,
        thinking_effort: ThinkingLevel,
    ) -> Path:
        model = model.replace("/", "_")
        cache_key = (
            model,
            self.task_name,
            instructions,
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
        model: str,
        instructions: list,
        thinking_effort: ThinkingLevel,
    ) -> str | None:
        query_cache_path = self.get_query_cache_path(
            model=model,
            instructions=instructions,
            thinking_effort=thinking_effort,
        )
        if query_cache_path.exists():
            return orjson.loads(query_cache_path.read_bytes())["output"]
        return None

    def save_to_cache(
        self,
        *,
        inputs: Inputs,
        model: str,
        instructions: list,
        thinking_effort: ThinkingLevel,
        output: str,
    ) -> None:
        query_cache_path = self.get_query_cache_path(
            model=model,
            instructions=instructions,
            thinking_effort=thinking_effort,
        )
        query_cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "inputs": inputs.model_dump(),
            "output": output,
            "model": model,
            "task_name": self.task_name,
            "thinking_effort": thinking_effort,
            "instructions": instructions,
        }
        with query_cache_path.open("wb") as f:
            f.write(orjson.dumps(data))


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).replace("’", "'").strip()


@dataclass
class CustomEvaluator(Evaluator[Inputs, ExpectedOutput, MetaData]):
    def evaluate(
        self, ctx: EvaluatorContext[Inputs, ExpectedOutput, MetaData]
    ) -> bool | EvaluationReason:
        expected_output = ctx.expected_output
        if not expected_output:
            raise RuntimeError("expected_output should not be null")

        output = typing.cast(str, ctx.output)
        output = output.strip()

        normalized_output = normalize(output)

        if expected_output.full_exact_match is not None:
            normalized_expected = normalize(expected_output.full_exact_match)
            # Split by words
            expected_words = normalized_expected.split(" ")
            actual_words = normalized_output.split(" ")
            diffs = list(Differ().compare(expected_words, actual_words))
            differences = [d for d in diffs if not d.startswith("  ")]
            if differences:
                return EvaluationReason(
                    value=False,
                    reason=f"differences: {differences}",
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
    chat_model = OpenAIChatModel(model, provider=OpenAIProvider())
    agent = Agent(
        chat_model,
        output_type=str,
        capabilities=[Thinking(effort=thinking_effort)],
    )
    llm_output_cache = ModelOutputCache(task_name=TASK_NAME)

    async def run_task(inputs: Inputs) -> str:
        instructions = get_instructions(inputs)

        if cached := llm_output_cache.check_cache(
            model=model,
            instructions=instructions,
            thinking_effort=thinking_effort,
        ):
            set_eval_attribute("cache_hit", True)
            return cached

        set_eval_attribute("cache_hit", False)
        increment_eval_metric("api_calls", 1)
        response = await agent.run(user_prompt=instructions)
        increment_eval_metric("tokens", response.usage.total_tokens)
        result = response.output
        llm_output_cache.save_to_cache(
            inputs=inputs,
            model=model,
            instructions=instructions,
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
