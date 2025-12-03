import typing
from dataclasses import dataclass

from openfoodfacts.utils.text import get_tag
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from .schemas import (
    CategoryPredictionInput,
    CategoryPredictionResponseModel,
    ExpectedResult,
    MetaData,
)


@dataclass
class CheckExtraction(Evaluator[CategoryPredictionInput, ExpectedResult, MetaData]):
    """Check that the predicted categories are correct."""

    def evaluate(
        self,
        ctx: EvaluatorContext[CategoryPredictionInput, ExpectedResult, MetaData],
    ) -> dict[str, EvaluationReason]:
        output = CategoryPredictionResponseModel.model_validate_json(
            typing.cast(str, ctx.output)
        )
        expected_output: ExpectedResult = typing.cast(
            ExpectedResult, ctx.expected_output
        )

        assertions = {}
        precise_match = False
        broader_match = False
        for predicted_category in output.categories:
            predicted_category_tag = (
                f"{predicted_category.language}:{get_tag(predicted_category.category)}"
            )
            if predicted_category_tag in expected_output.precise_category_patterns:
                precise_match = True
            elif predicted_category_tag in expected_output.broader_category_patterns:
                broader_match = True

        reason = (
            None
            if precise_match
            else (f"expected one of: {expected_output.precise_category_patterns}")
        )
        assertions["precise_category_match"] = EvaluationReason(
            value=precise_match,
            reason=reason,
        )

        broader_match = precise_match or broader_match
        reason = (
            None
            if broader_match
            else (f"expected one of: {expected_output.broader_category_patterns}")
        )
        assertions["broader_category_match"] = EvaluationReason(
            value=broader_match,
            reason=reason,
        )

        if expected_output.type is not None:
            assertions["type_match"] = EvaluationReason(
                value=output.type == expected_output.type,
                reason=(
                    None
                    if output.type == expected_output.type
                    else f"expected: {expected_output.type}, got: {output.type}"
                ),
            )
        return assertions
