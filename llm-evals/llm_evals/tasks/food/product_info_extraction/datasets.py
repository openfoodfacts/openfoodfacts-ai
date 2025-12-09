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
                ingredients=[
                    IngredientList(
                        language="en",
                        ingredients="Organic Beet Root Juice Powder.",
                        truncated=False,
                    )
                ],
                labels=[],
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
                brands=["Ginger People"],
                ingredients=[],
                labels=[],
            ),
        ),
    ],
    evaluators=[
        IsJson(),
        IsCorrectJsonSchema(pydantic_class=ProductInfoExtractionResponseModel),
        CheckExtraction(),
    ],
)
