from pydantic import BaseModel, Field


class IngredientList(BaseModel):
    language: str = Field(
        ..., description="Language (ISO 639-1 code) of the ingredient list"
    )
    ingredients: str = Field(..., description="Full ingredient list text")
    truncated: bool = Field(
        ..., description="Whether the ingredient list is truncated in the image"
    )


class SingleNutrientInfo(BaseModel):
    name: str = Field(..., description="Name of the nutrient, in English")
    value: str = Field(
        ..., description="Value of the nutrient, without unit (e.g., '10')"
    )
    unit: str = Field(..., description="Unit of the nutrient (e.g., 'g', 'mg')")


class NutritionInfo(BaseModel):
    per: str = Field(
        ...,
        description="For how much the nutrition info is provided. Most common values should be: '100g', 'serving', '100g_prepared', 'serving_prepared'. For 100ml, use '100g'. For 100ml prepared, use '100g_prepared'.",
    )
    nutrients: list[SingleNutrientInfo] = Field(
        ..., description="List of nutrients with their values"
    )
    truncated: bool = Field(
        ..., description="Whether the nutrition information is truncated in the image"
    )


class QuantityInfo(BaseModel):
    amount: str = Field(
        ..., description="Product quantity, with the unit (ex: '500 g')"
    )
    qualifier: str = Field(
        ...,
        description="Qualifier of the quantity, in English (e.g., 'net weight', 'drained weight',...)",
    )


class LabelInfo(BaseModel):
    name: str = Field(
        ...,
        description="Name of the label, in the original language. Example: 'USDA Organic', 'Label Rouge', ...)",
    )
    language: str | None = Field(
        ...,
        description="Language (ISO 639-1 code) of the label text, if any. If the label is just a logo, return null.",
    )
    text: str | None = Field(
        ...,
        description="Full label text, if any. If the label is just a logo, return null.",
    )


class ProductInfoExtractionResponseModel(BaseModel):
    brands: list[str] = Field(
        ...,
        description="List of brands of the product. If no brand is present, return an empty list. Example: ['Coca-Cola', 'Pepsi', 'Bjorg', 'Alpro']",
    )
    ingredients: list[IngredientList] = Field(
        ...,
        description="List of ingredient lists in different languages. If no ingredient list is present, return an empty list.",
    )
    labels: list[LabelInfo] = Field(
        ...,
        description="List of labels present on the packaging. By label, we mean a specific "
        "product claim (ex: 'low in fat'), or an official label, with a logo (ex: EU Organic, "
        "Label Rouge, PGI, USDA Organic,...).  The label can be a logo, or just text. If no label "
        "is present, return an empty list.",
    )
    nutrition: NutritionInfo | None = Field(
        default=None,
        description="Nutrition information of the product. If the information is not present, return null. "
        "If the information is partially present, only return the information visible on the image.",
    )
    quantity: QuantityInfo | None = Field(
        default=None,
        description="Quantity information of the product. If not present, return null.",
    )
    comment: str | None = Field(
        default=None,
        description="A free-text form, for any additional comment about the extraction, such as missing information or uncertainties.",
    )
