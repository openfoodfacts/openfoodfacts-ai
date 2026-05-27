import typing
from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from .schemas import ExpectedResult, Input, MetaData, PersonalInfoList


@dataclass
class CheckDetection(Evaluator[Input, ExpectedResult, MetaData]):
    """Check that the detected personal information are correct.

    Accuracy, precision, recall and f1-score are computed.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[Input, ExpectedResult, MetaData],
    ) -> dict[str, EvaluationReason | bool | float]:
        output = PersonalInfoList.model_validate_json(typing.cast(str, ctx.output))
        expected_output: ExpectedResult = typing.cast(
            ExpectedResult, ctx.expected_output
        )

        true_positives = []
        false_negatives = []
        for item in expected_output.items:
            # Find the best matching item
            match = next(
                (
                    x
                    for x in output.items
                    if x.type == item.type and x.value == item.value
                ),
                None,
            )
            if match:
                true_positives.append((item, match))
            else:
                false_negatives.append(item)
        false_positives = [
            item
            for item in output.items
            if item not in (tp[0] for tp in true_positives)
        ]
        precision = (
            (len(true_positives) / (len(true_positives) + len(false_positives)))
            if len(true_positives) + len(false_positives) > 0
            else None
        )
        recall = (
            (len(true_positives) / (len(true_positives) + len(false_negatives)))
            if len(true_positives) + len(false_negatives) > 0
            else None
        )
        accuracy = (
            (len(true_positives) / len(expected_output.items))
            if len(expected_output.items) > 0
            else None
        )

        if not len(expected_output.items):
            no_detections = int(len(output.items) == 0)
        else:
            no_detections = None
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "no_detections": no_detections,
        }
        return {k: v for k, v in metrics.items() if v is not None}
