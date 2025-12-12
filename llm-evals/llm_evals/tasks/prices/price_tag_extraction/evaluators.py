import math
import typing
from dataclasses import dataclass

from openfoodfacts.barcode import (
    calculate_check_digit,
    has_valid_check_digit,
    normalize_barcode,
)
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from .schemas import ExpectedResult, Label, MetaData, PriceTagExtractionInput


def barcode_is_valid(barcode: str) -> bool:
    return barcode.isnumeric() and len(barcode) >= 6 and has_valid_check_digit(barcode)


def barcode_fix_short_codes_from_usa(barcode: str) -> str:
    """
    Fix short barcodes from USA

    10 or 11 digits: pad them to 12 digits and calculate their check digit
    12 digits: add a leading zero

    This function is based on the fact that most of the short barcodes are from
    the USA, and that they can be converted to a valid EAN-13 barcode by
    padding them with zeros to the left, and adding a leading zero.

    :param barcode: the barcode to fix
    :return: the 13-digit fixed and valid barcode, or the original barcode if
    it cannot be fixed
    """
    if len(barcode) in (10, 11, 12):
        barcode_temp = barcode.zfill(12)
        barcode_check_digit = calculate_check_digit(barcode + "0")
        barcode_temp = barcode_temp + barcode_check_digit
        if barcode_is_valid(barcode_temp):
            return barcode_temp
    elif len(barcode) == 12:
        barcode_temp = "0" + barcode
        if barcode_is_valid(barcode_temp):
            return barcode_temp
    return barcode


def normalize_barcode_for_compare(barcode: str, currency: str) -> str:
    barcode = barcode.replace(" ", "")
    if (
        len(barcode) < 13
        and not barcode_is_valid(barcode)
        and currency.lower() == "usd"
    ):
        barcode = barcode_fix_short_codes_from_usa(barcode)
    return normalize_barcode(barcode)


@dataclass
class CheckExtraction(Evaluator[PriceTagExtractionInput, ExpectedResult, MetaData]):
    """Check that the extracted information from price tags is correct."""

    def evaluate(
        self,
        ctx: EvaluatorContext[PriceTagExtractionInput, ExpectedResult, MetaData],
    ) -> dict[str, EvaluationReason | bool]:
        output = Label.model_validate_json(typing.cast(str, ctx.output))
        expected_output: ExpectedResult = typing.cast(
            ExpectedResult, ctx.expected_output
        )

        evaluation = {
            "price": self.check_price(output, expected_output, metadata=ctx.metadata),
            "barcode": self.check_barcode(
                output, expected_output, metadata=ctx.metadata
            ),
        }
        return {k: v for k, v in evaluation.items() if v is not None}

    def check_price(
        self,
        output: Label,
        expected_output: ExpectedResult,
        metadata: MetaData,
    ) -> EvaluationReason | bool | None:
        selected_price = output.selected_price

        if expected_output.price is None:
            return EvaluationReason(
                value=selected_price is None, reason="No price expected"
            )

        tags = metadata.get("tags") or []
        if (
            "data-quality:price-truncated" in tags
            or "data-quality:price-unreadable" in tags
        ):
            return None

        if selected_price is None:
            return EvaluationReason(value=False, reason="No price extracted")

        match = math.isclose(selected_price.price, expected_output.price, abs_tol=0.01)

        if match:
            return True
        else:
            return EvaluationReason(
                value=False,
                reason=(
                    f"Expected price: {expected_output.price}, got: {selected_price}"
                ),
            )

    def check_barcode(
        self,
        output: Label,
        expected_output: ExpectedResult,
        metadata: MetaData,
    ) -> EvaluationReason | bool | None:
        tags = metadata.get("tags") or []
        if (
            expected_output.product_code is None
            or "data-quality:barcode-unreadable" in tags
            or "data-quality:barcode-truncated" in tags
        ):
            return None

        if output.barcode is None:
            return EvaluationReason(value=False, reason="No barcode extracted")

        currency = expected_output.currency or "eur"
        predicted_barcode = normalize_barcode_for_compare(
            output.barcode, currency=currency
        )

        if not predicted_barcode.isnumeric():
            return EvaluationReason(
                value=False,
                reason=f"Extracted barcode is not numeric: {output.barcode}",
            )

        match = predicted_barcode == normalize_barcode(expected_output.product_code)

        if match:
            return True
        else:
            return EvaluationReason(
                value=False,
                reason=(
                    f"Expected barcode: {expected_output.product_code}, got: {output.barcode}"
                ),
            )
