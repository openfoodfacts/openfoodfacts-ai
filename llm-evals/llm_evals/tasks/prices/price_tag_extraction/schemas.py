import enum
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, computed_field


class Unit(enum.StrEnum):
    KILOGRAM = "KILOGRAM"
    LITER = "LITER"
    UNIT = "UNIT"


class DiscountType(enum.StrEnum):
    QUANTITY = "QUANTITY"  # example: buy 1 get 1 free
    SALE = "SALE"  # example: 50% off
    SEASONAL = "SEASONAL"  # example: Christmas sale
    LOYALTY_PROGRAM = "LOYALTY_PROGRAM"  # example: 10% off for members
    EXPIRES_SOON = "EXPIRES_SOON"  # example: 30% off expiring soon
    PICK_IT_YOURSELF = "PICK_IT_YOURSELF"  # example: 5% off for pick-up
    SECOND_HAND = "SECOND_HAND"  # example: second hand books or clothes
    OTHER = "OTHER"
    NO_DISCOUNT = "NO_DISCOUNT"  # no discount applied


class SelectedPrice(BaseModel):
    price: float
    currency: str | None
    price_per: Unit
    price_is_discounted: bool = False
    price_without_discount: float | None
    discount_type: DiscountType = DiscountType.NO_DISCOUNT
    with_vat: bool = True


class LabelPrice(BaseModel):
    """One of the prices displayed on a price tag.

    For raw products (without barcode), if the price is indicated per weight
    but is not per kilogram (example: per 100g or per 500g), the price per
    kilogram should be calculated.
    """

    price: float = Field(..., description="Price in the local currency")
    with_vat: bool = Field(
        True,
        description="True if the price includes VAT (Value Added Tax), false otherwise. "
        "If there is no VAT in the country, set to false. In most cases, this should be set to true.",
    )
    currency: str | None = Field(
        ...,
        description="Currency of the price. Set to null if unknown. Examples: 'EUR', 'USD', 'GBP'",
    )
    price_per: Unit = Field(
        ...,
        description="Unit of the price. Can be one of: KILOGRAM: price is per kg, only if type = CATEGORY; "
        "LITER: price is per liter, only if type = CATEGORY; "
        "UNIT: price is per unit (available for both CATEGORY and PRODUCT types)",
    )
    price_is_discounted: bool = Field(
        False,
        description="true if this particular price entry is a discounted price, false otherwise",
    )
    discount_type: DiscountType = Field(
        DiscountType.NO_DISCOUNT,
        description="The type of discount applied to the price, if any. "
        "If no discount is applied, this should be set to NO_DISCOUNT. "
        "Possible discount types are: "
        " - QUANTITY: example: buy 1 get 1 free, "
        " - SALE: example: 50% off, "
        " - SEASONAL: example: Christmas sale, "
        " - LOYALTY_PROGRAM: example: 10% off for members, "
        " - EXPIRES_SOON: example: 30% off expiring soon, "
        " - PICK_IT_YOURSELF: example: 5% off for pick-up, "
        " - SECOND_HAND: example: second hand books or clothes, "
        " - OTHER: other types of discounts.",
    )
    uncertain: bool = Field(
        False,
        description="true if the price is uncertain: "
        "1) if the price is occluded (even partially), or blurred, "
        "2) if there is a reflection preventing accurate price reading, "
        "3) if the image quality is not good enough to read the price. "
        "Otherwise, the value must be false.",
    )


class Label(BaseModel):
    """A label (also called price tag) indicates in a store the price of the
    product and possibly other information such as the category, price per kg,
    origins, etc.

    We distinguish between two types of labels:

    - Labels for packaged products, with barcode (type: PRODUCT): these are
    products that have a barcode, which is usually displayed on the price tag.
    For this type of label, the category and origins should be set to null.
    - Raw products, without barcode (type: CATEGORY): these are usually fruits
    and vegetables, but it can also be any product sold per weight (kg, 100g,
    etc.). For this type of label, the category should be set to the
    corresponding category, the origins should be set to the countries of
    origin of the product (if indicated) and the barcode should be null.
    """

    type: Literal["PRODUCT", "CATEGORY"] | None = Field(
        description="The type of product the label is referring to. It should be "
        "`PRODUCT` for packaged products with barcode, and `CATEGORY` for raw "
        "products without barcode. If the type is unknown, this should be set to null.",
    )
    category: str | None = Field(
        description="The category of the product. If type=CATEGORY, this should be set to the "
        "category of the product, such as Apples, Bananas, Tomatoes, etc. The category must be in English, "
        "even if the category is displayed in another language on the price tag. "
        "The category must be the most precise category possible: for example, if the label says 'Red Onions', "
        "category should be 'Red onions', and not 'Onions'. "
        "If more than one category is present on the price tag, only the first category should be considered, and "
        "`has_multiple_categories` should be set to true. "
        "If unknown, or if TYPE=PRODUCT, this should be set to null.",
    )
    has_multiple_categories: bool | None = Field(
        description="If type is CATEGORY: if more than one category is present on the price tag, this should be true, "
        "false otherwise. If type is PRODUCT: it should be null",
    )
    prices: list[LabelPrice] = Field(
        description="All prices found on the label. Depending on the type of "
        "price tag, there can be multiple prices displayed. "
        "For packaged products (type=PRODUCT), the price per unit is the most common one. "
        "The price per kg is also often included. "
        "For raw products (type=CATEGORY), the price per kg is the most common one. "
        "There can also be a discount applied to the price. In such a case, both "
        "the original price and the discounted price should be included in the list.",
    )
    origins: list[str] | None = Field(
        description="The countries of origin of the product, in English. "
        "If type=PRODUCT, this should be set to null. If type=CATEGORY, this should "
        """be set to the countries of origin of the product, such as ["France"], ["Italy"], """
        """["Spain"], etc. """
        "Most of the time, there is only one country of origin indicated on the label, "
        "but sometimes there can be multiple countries (for example: 'France and Spain'), in "
        "which case all countries should be included in the list. "
        "If type=CATEGORY and the origin is unknown, it should be set to null.",
    )
    organic: bool | None = Field(
        description="true if the product is organic, false otherwise. If true, "
        "there should be evidence on the label suggesting that the product is "
        "organic, such as the EU organic logo or other recognized organic certifications. If the organic status is unknown, "
        "it should be set to null.",
    )
    barcode: str | None = Field(
        description="The barcode of the product, if available. "
        "The barcode is usually a number with 13 (EAN13) or 8 (EAN8) digits. You should "
        "*NOT* try to decode the barcode stripe (also called modules), but use the "
        "barcode number displayed on the label. "
        "If type=CATEGORY, this should be null.",
    )
    product_name: str = Field(
        description="The name of the product, as displayed on the label. "
        "For raw products (type=CATEGORY), this is usually the name of the fruit or vegetable. "
        "For products with barcode (type=PRODUCT), it usually includes the brand, a short "
        "description of the product, and optionally the quantity. "
        "If no product name is displayed on the label, this should be an empty string. "
        "examples: 'NOCCIOLATA BIO 650G', 'Simpson Donuts', 'GERBLE BISCUIT PIST ABRICOT160G', "
        "'Radis Blanc', 'Concombre lisse', 'Courget Butternut', 'Tomatoes', 'Organic Bananas'",
    )
    uncertain_barcode_or_product_name: bool = Field(
        description="true if the barcode (for type=PRODUCT) or category (for TYPE=CATEGORY) is uncertain: "
        "1) if the barcode (respectively the product name) is occluded (even partially), or blurred,"
        "2) if there is a reflection preventing accurate reading, "
        "3) if the image quality is not good enough to read the barcode (respectively the product name). "
        "Otherwise, the value must be false.",
    )
    is_price_tag: bool = Field(
        description="indicates whether the image shows a physical price tag attached to a product. "
        "For example, if the image seems to come from a receipt or a catalogue "
        "(or is a random image), this should be set to false.",
    )

    @computed_field
    @property
    def selected_price(self) -> SelectedPrice | None:
        """From all individual price reference on the price tag, construct a
        Price ready to be added to Open Prices.
        """
        if not self.prices:
            return None

        # sort prices to have prices with VAT first
        sorted_prices = sorted(self.prices, key=lambda p: p.with_vat, reverse=True)

        price_grouped_by_per: dict[Unit, list[LabelPrice]] = {unit: [] for unit in Unit}
        for price in sorted_prices:
            # Convert price_per to Unit.KILOGRAM if it is LITER
            price_per = (
                price.price_per if price.price_per != Unit.LITER else Unit.KILOGRAM
            )
            price_grouped_by_per[price_per].append(price)

        if self.type == "CATEGORY":
            # We first consider the price per KILOGRAM for raw products then
            # per UNIT
            selected_units = [Unit.KILOGRAM, Unit.UNIT]
        else:
            # We only consider the price per UNIT for packaged products.
            selected_units = [Unit.UNIT]

        for selected_unit in selected_units:
            prices_per_selected_unit = price_grouped_by_per[selected_unit]
            # Get the first occurence of a price with the selected unit and
            # without discount. If there are one price with VAT and one without
            # VAT, we take the one with VAT (thanks to sorting).
            no_discount_price = next(
                (
                    p
                    for p in prices_per_selected_unit
                    if p.discount_type is DiscountType.NO_DISCOUNT
                ),
                None,
            )
            discounted_price = next(
                (
                    p
                    for p in prices_per_selected_unit
                    # We only consider the discount types SALE and SEASONAL as
                    # these are the only ones that applies to any client.
                    if p.discount_type in (DiscountType.SALE, DiscountType.SEASONAL)
                ),
                None,
            )
            if discounted_price:
                # There is a discounted price displayed on the price tag.

                # We search again for a price without discount, but with the
                # same VAT status as the discounted price (otherwise the two
                # prices cannot be compared).
                no_discount_price = next(
                    (
                        p
                        for p in prices_per_selected_unit
                        if p.discount_type is DiscountType.NO_DISCOUNT
                        and p.with_vat == discounted_price.with_vat
                    ),
                    None,
                )
                return SelectedPrice(
                    price=discounted_price.price,
                    with_vat=discounted_price.with_vat,
                    currency=discounted_price.currency,
                    price_per=selected_unit,
                    price_is_discounted=True,
                    # If we didn't find a price without discount, we set the
                    # price_without_discount to null (as it's unknown).
                    price_without_discount=(
                        no_discount_price.price if no_discount_price else None
                    ),
                    discount_type=discounted_price.discount_type,
                )
            elif no_discount_price:
                # This is a regular price, without discount
                return SelectedPrice(
                    price=no_discount_price.price,
                    with_vat=no_discount_price.with_vat,
                    currency=no_discount_price.currency,
                    price_per=selected_unit,
                    price_is_discounted=False,
                    price_without_discount=None,
                    discount_type=DiscountType.NO_DISCOUNT,
                )

        return None


class ExpectedResult(BaseModel):
    type: Literal["PRODUCT", "CATEGORY"]
    product_code: str | None = None
    category: list[str] | None = None
    labels_tags: list[str] | None = None
    origins_tags: list[str] | None = None
    price: float | None = None
    price_is_discounted: bool | None = None
    price_without_discount: float | None = None
    discount_type: DiscountType | None = None
    price_per: Unit | None = None
    currency: str | None = None


class PriceTagExtractionInput(TypedDict):
    image_urls: list[str]


class MetaData(TypedDict):
    price_tag_id: int
    tags: list[str] | None
