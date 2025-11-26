import json
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class IsJson(Evaluator[object, object, object]):
    """Check that the output is a valid JSON string."""

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> bool | dict[str, bool]:
        output = ctx.output

        if not isinstance(output, str):
            return False

        try:
            json.loads(output)
            is_valid_json = True
        except (ValueError, TypeError):
            is_valid_json = False
        return is_valid_json


@dataclass
class IsCorrectJsonSchema(Evaluator):
    """Check that the output conforms to the given Pydantic class."""

    pydantic_class: type[BaseModel]

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> bool | dict[str, bool]:
        output = ctx.output

        if not isinstance(output, str):
            return False

        try:
            parsed_output = json.loads(output)
        except (ValueError, TypeError):
            return False

        try:
            self.pydantic_class.model_validate(parsed_output)
            return True
        except Exception:
            return False
