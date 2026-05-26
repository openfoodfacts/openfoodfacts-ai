from typing import Literal
from pydantic import BaseModel, Field


class IngredientList(BaseModel):
    language: str = Field(
        description="Language (ISO 639-1 code) of the ingredient list"
    )
    ingredients: str = Field(
        description="Full ingredient list text. The ingredient text should not include "
        "any prefix such as 'Ingredients:' or 'Ingredients list:'. It should only "
        "include the actual list of ingredients. Allergen mentions should be included "
        "in the ingredient list only if they are present just after the ingredient list, "
        "and not in a separate allergen statement. Do *NOT* reconstruct missing parts of "
        "the ingredient list if it is truncated or occluded on the image, just return the "
        "visible part of the ingredient list.",
    )
    truncated: bool = Field(
        description="Whether the ingredient list is truncated or occluded on the image",
    )


class NutritionNutrientInput(BaseModel):
    unit: str = Field(
        description="The unit of the value for the product. "
        "The possible values depends on the nutrient:\n"
        "* `g` for grams\n"
        "* `kcal` for kilocalories\n"
        "* `kj` for kilojoules\n"
        "* `mg` for milligrams\n"
        "* `μg` for micrograms\n"
        "* `cl` for centiliters\n"
        "* `ml` for mililiters\n"
        "* `dv` for recommended daily intakes (aka Dietary Reference Intake)\n"
        "* `% vol` for alcohol vol per 100 ml\n"
    )
    value: float | None = Field(
        description="A normalized float value for the quantity, computed from "
        "`value_string` if it exists."
    )
    value_string: str | None = Field(
        description="A string representing the value of the quantity. This is the raw "
        "value as extracted from the image, without any normalization."
    )
    modifier: Literal["<", "<=", "~", ">=", ">"] | None = Field(
        description="Modifier of the nutrient value, if any. For example, if the label "
        "states 'less than 0.5g' or '< 0.5g', then the value would be 0.5 and the "
        "modifier would be '<'.",
    )


class NutritionInputSet(BaseModel):
    preparation: Literal["as_sold", "prepared"] = Field(
        description="Indicates whether the nutrition values refer to the product "
        "*as_sold* or *prepared*. The preparation state affects nutrient values."
    )
    per: Literal["serving", "100g", "100ml"] = Field(
        description="The nutrition data on the package can be per serving, per 100g "
        "or per 100ml. This is essential to understand if values in the `nutrients` "
        "object apply for a serving, for 100g or for 100ml."
    )
    per_quantity: int | None = Field(
        description="The nutrition data on the package can be per serving, per 100g or "
        "per 100ml. When the data is given per serving, the actual quantity that "
        "defines one serving may vary between products and is stored in this field. "
        "This is essential to understand to which quantity values in `nutrients` apply "
        "for. For example, if the label states 'per 250g', then this field should be "
        "250. If the label states 'per serving' and the quantity of one serving is not "
        "specified, then this field should be null.",
    )
    per_unit: Literal["g", "ml"] | None = Field(
        description="The nutrition data on the package can be per serving, per 100g or "
        "per 100ml. When the data is given per serving, the actual unit that defines "
        "one serving may vary between products and is stored in this field. "
        "This is essential to understand to which quantity values in `nutrients` apply "
        "for. For example, if the label states 'per 250g', then this field should be "
        "'g'. If the label states 'per serving' and the quantity of one serving is not "
        "specified, then this field should be null.",
    )
    nutrients: dict[str, NutritionNutrientInput] = Field(
        default_factory=dict,
        description="For all detected nutrient, this is a mapping from nutrient name "
        "to its value. No conversion between serving size and 100g/100ml values should "
        "be done, the value should be exactly as extracted from the image. "
        "The nutrient name must be in English. Use '-' instead of '_' "
        "to separate nutrient name (ex: `saturated-fat`, not `saturated_fat`). "
        "Example of nutrient names:\n"
        "* `energy-kcal`\n"
        "* `energy-kj`\n"
        "* `carbohydrates`\n"
        "* `saturated-fat`\n"
        "* `sugars`\n"
        "* `salt`\n"
        "* `sodium`\n"
        "* `fiber`\n"
        "* `proteins`\n",
    )


class NutritionInfo(BaseModel):
    input_sets: list[NutritionInputSet] = Field(
        description="List of nutrition input sets, for different preparation states and/or "
        "different 'per' values. Nutrition information are often provided either per 100g and per "
        "serving, and for the product 'as sold' or for the prepared product (ex: instant coffee). "
        "Each combination is one input set. Extract all nutrition input sets from the image.",
    )
    truncated: bool = Field(
        description="Whether the nutrition information is truncated or occluded on the image",
    )


class QuantityInfo(BaseModel):
    amount: str = Field(description="Product quantity, with the unit (ex: '500 g')")
    qualifier: str | None = Field(
        description="Qualifier of the quantity, in English (e.g., 'net weight', "
        "'drained weight',...). If the quantity has no qualifier, return null.",
    )


class LabelInfo(BaseModel):
    name: str = Field(
        description="Name of the label, in the original language. Example: 'USDA Organic', 'Label Rouge', ...)",
    )
    language: str | None = Field(
        description="Language (ISO 639-1 code) of the label text, if any. If the label is just a logo, return null.",
    )
    text: str | None = Field(
        description="Full label text, if any. If the label is just a logo, return null.",
    )


class ProductNameInfo(BaseModel):
    name: str = Field(
        description="A name for this product for the language specified in the `lang` field."
    )
    lang: str = Field(
        description="Language (ISO 639-1 code) associated with the product name"
    )


class ProductInfoExtractionResponseModel(BaseModel):
    product_names: list[ProductNameInfo] = Field(
        description="Name of the product in each language present on the packaging. "
        "The product name can be either extracted from the image, or generated if "
        "we're confident of what the product is. If no product name could be reliably "
        "generated, return an empty list.",
    )
    main_lang: str | None = Field(
        description="The main language of the product, if it can be determined. The "
        "main language is the language in which the most information on the packaging "
        "is written. If it cannot be determined, return null.",
    )
    brands: list[str] = Field(
        description="List of brands of the product. If no brand is present, return an empty list. Examples: 'Coca-Cola', 'Pepsi', 'Bjorg', 'Alpro'",
    )
    ingredients: list[IngredientList] = Field(
        description="List of ingredient lists in different languages. If no ingredient list is present, return an empty list.",
    )
    labels: list[LabelInfo] = Field(
        description="List of labels present on the packaging. By label, we mean a specific "
        "product claim (ex: 'low in fat'), or an official label, with a logo (ex: EU Organic, "
        "Label Rouge, PGI, USDA Organic,...).  The label can be a logo, or just text. If no label "
        "is present, return an empty list. The list of possible allergens should *NOT* be included in this field.",
    )
    nutrition: NutritionInfo | None = Field(
        description="Nutrition information displayed on the image. If the information is not present, return null. "
        "If the information is partially present, only return the information visible on the image.",
    )
    quantity: QuantityInfo | None = Field(
        description="Quantity information of the product. If not present, return null.",
    )
    comment: str | None = Field(
        description="A free-text form, for any additional comment about the extraction, such as missing information or uncertainties.",
    )
