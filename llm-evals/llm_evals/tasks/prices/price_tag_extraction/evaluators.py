import itertools
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


def generate_pluralization_forms_single_word(category: str) -> set[str]:
    """Generate possible plural and singular forms of a category string
    (single word). The input category string is assumed to be in plural form.

    This is a simple heuristic that handles common pluralization patterns in
    English.

    :param category: the category string
    :return: a list of possible singular forms
    """
    forms = set()
    forms.add(category)

    if category.lower() in ("tomatoes", "potatoes"):
        forms.add(category[:-2])
    elif category.endswith("ies"):
        forms.add(category[:-3] + "y")
    elif category.endswith("ves"):
        forms.add(category[:-3] + "f")
        forms.add(category[:-3] + "fe")
    elif category.endswith("s"):
        forms.add(category[:-1])

    return forms


def generate_pluralization_forms(category: str) -> set[str]:
    """Generate possible singular and plural forms of a category string.
    The category string can contain multiple words, and is assumed to be in
    plural form.

    This function splits the category string into words and generates singular
    forms for each word using common pluralization patterns in English.

    We combine all possible singular forms of each word to generate all
    possible singular forms of the entire category string.

    The generated singular form may not always be correct, but as we are using
    it to compare expected and generated categories, it should be sufficient.

    :param category: the category string
    :return: a list of possible singular forms
    """
    words = category.split()
    words_with_all_forms = []
    forms = set()

    for i in range(len(words)):
        word_singular_forms = generate_pluralization_forms_single_word(words[i])
        words_with_all_forms.append(word_singular_forms)

    for cartesian_product in itertools.product(*words_with_all_forms):
        forms.add(" ".join(cartesian_product))
    return forms


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
            "category": self.check_category(
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
        tags = metadata.get("tags") or []
        if (
            "data-quality:price-truncated" in tags
            or "data-quality:price-unreadable" in tags
        ):
            return None

        if expected_output.price is None:
            return EvaluationReason(
                value=selected_price is None, reason="No price expected"
            )

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
            "data-quality:barcode-unreadable" in tags
            or "data-quality:barcode-truncated" in tags
            or "with-internal-barcode" in tags
            or "type:category" in tags
        ):
            return None

        if expected_output.product_code is None:
            return (
                True
                if not output.barcode
                else EvaluationReason(
                    value=False,
                    reason="barcode extracted while no barcode was expected",
                )
            )

        # We know that expected product_code is not null now
        if not output.barcode:
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

    def check_category(
        self,
        output: Label,
        expected_output: ExpectedResult,
        metadata: MetaData,
    ) -> EvaluationReason | bool | None:
        if expected_output.type != "CATEGORY":
            if output.category:
                return EvaluationReason(
                    value=False,
                    reason="Category extracted while not expected",
                )
            return None

        tags = metadata.get("tags") or []
        if "data-quality:category-unreadable" in tags:
            return None

        if expected_output.category:
            if not output.category:
                return EvaluationReason(value=False, reason="No category extracted")

            expected_categories_set: set[str] = set()
            for category in expected_output.category:
                forms = generate_pluralization_forms(category)
                expected_categories_set.update(form.lower() for form in forms)
            output_category_set = generate_pluralization_forms(output.category.lower())

            if output_category_set & expected_categories_set:
                return True
            else:
                return EvaluationReason(
                    value=False,
                    reason=(
                        f"Expected categories: {expected_output.category}, got: {output.category}"
                    ),
                )

        return None
