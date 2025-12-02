from typing import TypedDict

from pydantic import BaseModel, Field


class CategoryInfo(BaseModel):
    category: str = Field(..., description="Name of the category of the product.")
    language: str | None = Field(
        ..., description="Language (ISO 639-1 code) associated with ."
    )


class CategoryPredictionResponseModel(BaseModel):
    categories: list[CategoryInfo] = Field(
        ...,
        description="List of categories associated with product. The most precise category should be returned first, then broader categories if possible. "
        "Prefer categories in English, unless there there are no good translations in English available. Don't return duplicate categories in different languages. "
        "If no category could not be determined, return an empty list.",
    )
    explanation: str = Field(
        ...,
        description="Explanation of the reasoning behind the category prediction.",
    )


class ExpectedResult(BaseModel):
    precise_category_patterns: list[str] = Field(
        ...,
        description="Most precise category. Several candidates can be listed, for comparison with the predicted category.",
    )
    broader_category_patterns: list[str] = Field(
        ...,
        description="Broader categories that also apply to the product. Several candidates can be listed, for comparison with the predicted categories.",
    )


class MetaData(TypedDict):
    barcode: int
    tags: list[str] | None


class CategoryPredictionInput(TypedDict):
    image_urls: list[str]
