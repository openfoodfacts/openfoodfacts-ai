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

import re
import typing
from dataclasses import dataclass
from difflib import Differ
from pathlib import Path

import orjson
import typer
from deepdiff import DeepHash
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ImageUrl, PromptedOutput
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


def normalize(s: str) -> str:
    return (
        # Replace all whitespace characters (including tabs, newlines, etc.) with a single space
        re.sub(r"\s+", " ", s)
        # Normalize quotes
        .replace("’", "'")
        # Remove leading/trailing whitespace
        .strip()
    )


def get_diff(s1: str, s2: str):
    expected_lines = s1.split(" ")
    actual_lines = s2.split(" ")
    diffs = list(Differ().compare(expected_lines, actual_lines))
    # Filter to only lines that differ
    return "\n".join([d for d in diffs if not d.startswith("  ")])


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
                            reason=f"truncated mismatch: output={output_list.truncated}, expected={expected_list.truncated}",
                        )

                    if output_list.truncated:
                        continue
                    normalized_output = normalize(output_list.ingredients)
                    normalized_expected = normalize(expected_list.ingredients)
                    if normalized_output != normalized_expected:
                        diff = get_diff(normalized_output, normalized_expected)
                        return EvaluationReason(
                            value=False,
                            reason=f"ingredients mismatch\ndiff:\n{diff}\noutput:\n{output_list.ingredients}\nexpected:\n{expected_list.ingredients}",
                        )
        return True


dataset = Dataset[Inputs, ExpectedOutput, MetaData](
    name=TASK_NAME,
    cases=[
        Case(
            name="8480000590909",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/848/000/059/0909/ingredients_es.15.400.jpg"
            ),
            expected_output=Output(
                ingredient_lists=[
                    IngredientList(
                        language="es",
                        ingredients=(
                            "Carne de cerdo, tocino, lactosa, sal, "
                            "proteínas de la leche, dextrosa, especias, aromas, "
                            "aromas de humo, estabilizantes (E-450, E-451), "
                            "antioxidante (E-301), conservadores (E-250, E-252), "
                            "colorante (E-120) y extracto de pimentón."
                        ),
                        truncated=False,
                    )
                ]
            ),
        ),
        Case(
            name="3481910513854",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/348/191/051/3854/ingredients_fr.20.full.jpg"
            ),
            expected_output=Output(
                ingredient_lists=[
                    IngredientList(
                        language="fr",
                        # Google Cloud OCR was missing the ";" after "carmins"
                        ingredients=(
                            "viande de bœuf 86%, eau, sel, dextrose, épices et plante aromatique, arômes naturels, acidifiants : lactate de potassium, acétates de sodium ; antioxydants : acide ascorbique, ascorbate de sodium ; colorant : carmins ; enveloppe : boyau naturel de mouton. Traces éventuelles de moutarde, gluten, pistache."
                        ),
                        truncated=False,
                    )
                ]
            ),
            metadata=MetaData(tags=["google-cloud-error"]),
        ),
        Case(
            name="3594800001116",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/359/480/000/1116/ingredients_fr.7.full.jpg"
            ),
            expected_output=Output(
                ingredient_lists=[
                    IngredientList(
                        language="fr",
                        ingredients=(
                            "Farine de BLE {(GLUTEN, agent de traitement de la farine: acide ascorbique, enzymes : xylanase (BLE), alpha-amylase (BLE)}, vin blanc (18,1%) (SULFITES), huile d'ARACHIDE, sucre, poudre levante (diphosphate disodique E450i, carbonate acide de sodium E500ii, farine de BLE)."
                        ),
                        truncated=False,
                    )
                ]
            ),
        ),
        Case(
            name="5410046001766",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/541/004/600/1766/ingredients_fr.7.full.jpg"
            ),
            expected_output=Output(
                ingredient_lists=[
                    IngredientList(
                        language="fr",
                        ingredients=(
                            "fruits: fraises 37%; rhubarbe 25%, sucre, gélifiant: pectines, correcteurs d'acidité: acide citrique; citrates de calcium, jus de cassis concentré, antioxydant: acide ascorbique. Sans édulcorant ni conservateur."
                        ),
                        truncated=False,
                    ),
                    IngredientList(
                        language="nl",
                        # Google Cloud OCR replaced "calciumcitraten" by "calciumcitrate"
                        ingredients=(
                            "vruchten: aardbeien 37%; rabarber 25%, suiker, geleermiddel: pectinen, zuurteregelaars: citroenzuur; calciumcitraten, zwarte bessap concentraat, antioxidant: ascorbinezuur. Zonder zoetstof noch conserveermiddel."
                        ),
                        truncated=False,
                    ),
                    IngredientList(
                        language="de",
                        # Google Cloud OCR was missing the ";" after "Erdbeeren 37%"
                        ingredients=(
                            "Früchte: Erdbeeren 37%; Rhabarber 25%, Zucker, Geliermittel: Pektine, Säureregulatoren: Citronensäure; Calciumcitrate, schwarzes Johannisbeerensaft-konzentrat, Antioxidationsmittel: Ascorbinsäure. Ohne Süßungsmittel und Konservierungsstoffe."
                        ),
                        truncated=False,
                    ),
                ]
            ),
            metadata=MetaData(tags=["google-cloud-error"]),
        ),
        Case(
            name="3392590205277",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/339/259/020/5277/ingredients_fr.11.full.jpg"
            ),
            expected_output=Output(
                ingredient_lists=[
                    IngredientList(
                        language="fr",
                        # 3 google cloud errors: "euf" instead of "œuf", missing ":" after "MIX POUR FRANGIPANE", "blé française" instead of "blé française*"
                        ingredients=(
                            "2 PÂTES FEUILLETÉES PUR BEURRE PRÊTES À DÉROULER : Farine de blé française* (54%), eau, beurre concentré origine France (19%), alcool éthylique, sel, levure désactivée, jus de citron concentré. *Ingrédient issu du commerce équitable français Agri-Éthique. MIX POUR FRANGIPANE : Sucre, farine de blé, amandes en poudre (12%), arôme. Peut contenir des traces d'œuf, de soja, de sésame et de fruits à coque."
                        ),
                        truncated=False,
                    ),
                ]
            ),
            metadata=MetaData(tags=["google-cloud-error"]),
        ),
        Case(
            name="5410046000028",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/541/004/600/0028/ingredients_fr.7.full.jpg"
            ),
            # We just check here that all ingredient lists are marked as truncated
            expected_output=Output(
                ingredient_lists=[
                    IngredientList(
                        language="fr",
                        # No need to add ingredients as the field is truncated.
                        ingredients="",
                        truncated=True,
                    ),
                    IngredientList(
                        language="nl",
                        ingredients="",
                        truncated=True,
                    ),
                    IngredientList(
                        language="de",
                        ingredients="",
                        truncated=True,
                    ),
                ]
            ),
        ),
        Case(
            name="3760275810168",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/376/027/581/0168/ingredients_fr.23.400.jpg"
            ),
            expected_output=Output(
                ingredient_lists=[
                    IngredientList(
                        language="fr",
                        # Google Cloud OCR errors:
                        # - "ceufs" instead of "oeufs"
                        # - "40%;" instead of "40% :"
                        # - "conser - vateur" instead of "conser - vateur"
                        # - "mono et. diglycérides" instead of "mono et diglycérides"
                        ingredients=(
                            "Pâtes fraîches à l'oeuf farcies à la viande.\n"
                            'Ingrédients des pâtes (60%) : Farine de blé tendre type "00", semoule de blé dur, oeufs mixte pasteurisé (17%), eau, sel. Ingrédients de la farce (40%) : Préparation de porc et de boeuf cuits 37% (porc 79%, boeuf 15%, sel, épices, arômes, antioxydant : ascorbate de sodium, conservateur : nitrite de sodium), chapelure (farine de blé tendre, eau, sel), eau, sel, huile de tournesol, flocons de pommes de terre (pommes de terre, émulsifiants : mono et diglycérides d\'acides gras), farine de blé tendre de type "00", fromage râpé (lait, sel, présure), fibre de bambou, vin rouge (sulfites), plantes aromatiques, arômes, épices. Peut contenir des traces de soja et de noix.'
                        ),
                        truncated=False,
                    ),
                ],
            ),
            metadata=MetaData(tags=["google-cloud-error"]),
        ),
        Case(
            name="3230140005031",
            inputs=Inputs(
                image_url="https://images.openfoodfacts.org/images/products/323/014/000/5031/ingredients_fr.5.full.jpg"
            ),
            expected_output=Output(
                ingredient_lists=[
                    IngredientList(
                        language="fr",
                        # Google Cloud OCR errors:
                        # - "Comichons" instead of "Cornichons"
                        # - "grains coriandre" instead of "graines de coriandre"
                        # - "aromes" instead of "arômes"
                        ingredients=(
                            "Cornichons (dont conservateur : disulfite de potassium), eau, vinaigre d'alcool, oignons, sel, estragon, graines de moutarde jaune, graines de coriandre, arômes naturels, affermissant : chlorure de calcium."
                        ),
                        truncated=False,
                    ),
                    IngredientList(
                        language="en",
                        ingredients=(
                            "Gherkins (preservative : potassium metabisulphite), water, vinegar, onions, salt, tarragon, yellow mustard seeds, coriander seeds, natural flavours, firming agent : calcium chloride."
                        ),
                        truncated=False,
                    ),
                    IngredientList(
                        language="de",
                        # Google Cloud OCR errors:
                        # - missing "(" before "Konservierungsmittel"
                        # - missing "," after "Branntweinessig"
                        # note: "Festigunsmittel" was incorrectly written "Festigungsmittel" on the packaging
                        ingredients=(
                            "Essiggurken (Konservierungsmittel : Kaliummetabisulfit), Wasser, Branntweinessig, Silberzwiebeln, Salz, Estragon, Senfkörner, Koriadersamen, natürliche Aromen, Festigunsmittel : Calciumchlorid."
                        ),
                        truncated=False,
                    ),
                ],
            ),
            metadata=MetaData(tags=["google-cloud-error"]),
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
