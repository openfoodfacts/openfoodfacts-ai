from pydantic_evals import Case, Dataset

from llm_evals.evaluators import IsCorrectJsonSchema, IsJson

from .evaluators import CheckExtraction
from .schemas import IngredientList, ProductInfoExtractionResponseModel

dataset = Dataset(
    cases=[
        Case(
            inputs={
                "image_urls": [
                    "https://images.openfoodfacts.org/images/products/085/001/561/6501/2.jpg"
                ]
            },
            expected_output=ProductInfoExtractionResponseModel(
                brands=[],
                product_names=[],
                ingredients=[
                    IngredientList(
                        language="en",
                        ingredients="Organic Beet Root Juice Powder.",
                        truncated=False,
                    )
                ],
                nutrition=None,
                quantity=None,
                comment=None,
                labels=[],
                main_lang=None,
            ),
            metadata={"country": "us", "langs": ["en"]},
        ),
        Case(
            inputs={
                "image_urls": [
                    "https://images.openfoodfacts.org/images/products/400/937/115/3199/1.jpg"
                ]
            },
            expected_output=ProductInfoExtractionResponseModel(
                product_names=[],
                brands=["Ginger People"],
                ingredients=[],
                labels=[],
                nutrition=None,
                quantity=None,
                comment=None,
                main_lang=None,
            ),
        ),
    ],
    evaluators=[
        IsJson(),
        IsCorrectJsonSchema(pydantic_class=ProductInfoExtractionResponseModel),
        CheckExtraction(),
    ],
)
