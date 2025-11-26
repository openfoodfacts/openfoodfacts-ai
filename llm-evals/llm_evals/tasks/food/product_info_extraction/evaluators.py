import typing
from dataclasses import dataclass

from openfoodfacts.utils.text import get_tag
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from .schemas import ProductInfoExtractionResponseModel


@dataclass
class CheckExtraction(Evaluator):
    """Check that the extracted ingredients list is correct."""

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> dict[str, EvaluationReason | bool]:
        output = ProductInfoExtractionResponseModel.model_validate_json(
            typing.cast(str, ctx.output)
        )
        expected_output: ProductInfoExtractionResponseModel = typing.cast(
            ProductInfoExtractionResponseModel, ctx.expected_output
        )
        return {
            "ingredients": self.check_ingredients(output, expected_output),
            "brands": self.check_brands(output, expected_output),
        }

    def check_ingredients(
        self,
        output: ProductInfoExtractionResponseModel,
        expected_output: ProductInfoExtractionResponseModel,
    ) -> EvaluationReason | bool:
        if not expected_output.ingredients:
            return EvaluationReason(
                value=len(output.ingredients) == 0, reason="No ingredients expected"
            )

        if not output.ingredients:
            return EvaluationReason(value=False, reason="No ingredients extracted")

        expected_ingredients = (
            expected_output.ingredients[0].ingredients.lower().strip()
        )
        output_ingredients = output.ingredients[0].ingredients.lower().strip()
        match = output_ingredients.startswith(expected_ingredients)
        if match:
            return True
        else:
            return EvaluationReason(
                value=False,
                reason=(
                    f"Extracted ingredients do not match expected.\n"
                    f"Expected: {expected_ingredients}\n"
                    f"Got: {output_ingredients}"
                ),
            )

    def check_brands(
        self,
        output: ProductInfoExtractionResponseModel,
        expected_output: ProductInfoExtractionResponseModel,
    ) -> bool:
        brands = [get_tag(brand) for brand in output.brands]
        expected_brands = [get_tag(brand) for brand in expected_output.brands]

        return set(brands) == set(expected_brands)
