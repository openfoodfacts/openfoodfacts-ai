from pydantic_evals import Case, Dataset

from llm_evals.evaluators import IsCorrectJsonSchema, IsJson

from .schemas import DiscountType, ExpectedResult, Label

dataset = Dataset(
    cases=[
        # In progress - to be filled with real data
        Case(
            inputs={
                "price_tag_url": "https://prices.openfoodfacts.org/api/v1/price-tags/100"
            },
            metadata={"country": "fr"},
            expected_output=ExpectedResult(
                type="PRODUCT",
                product_price_per_unit_with_discount=9.99,
                product_price_per_unit_without_discount=12.99,
                product_price_per_unit_discount_type=DiscountType.SALE,
            ),
        )
    ],
    evaluators=[
        IsJson(),
        IsCorrectJsonSchema(pydantic_class=Label),
    ],
)
