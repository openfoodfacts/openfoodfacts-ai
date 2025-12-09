import enum
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, computed_field


class RawCategory(enum.StrEnum):
    APPLES = "en:apples"
    APRICOTS = "en:apricots"
    ARTICHOKES = "en:artichokes"
    ASPARAGUS = "en:asparagus"
    AUBERGINES = "en:aubergines"
    AVOCADOS = "en:avocados"
    BANANAS = "en:bananas"
    BEETROOT = "en:beetroot"
    BERRIES = "en:berries"
    BLACKBERRIES = "en:blackberries"
    BLUEBERRIES = "en:blueberries"
    BOK_CHOY = "en:bok-choy"
    BROCCOLI = "en:broccoli"
    CABBAGES = "en:cabbages"
    CARROTS = "en:carrots"
    CAULIFLOWERS = "en:cauliflowers"
    CELERY = "en:celery"
    CELERIAC = "en:celeriac"
    CELERY_STALK = "en:celery-stalk"
    CEP_MUSHROOMS = "en:cep-mushrooms"
    CHANTERELLES = "en:chanterelles"
    CHARDS = "en:chards"
    CHERRIES = "en:cherries"
    CHERRY_TOMATOES = "en:cherry-tomatoes"
    CHICKPEAS = "en:chickpeas"
    CHIVES = "en:chives"
    CLEMENTINES = "en:clementines"
    COCONUTS = "en:coconuts"
    CRANBERRIES = "en:cranberries"
    CUCUMBERS = "en:cucumbers"
    DATES = "en:dates"
    ENDIVES = "en:endives"
    FENNEL_BULBS = "en:fennel-bulbs"
    FIGS = "en:figs"
    GARLIC = "en:garlic"
    GINGER = "en:ginger"
    GRAPEFRUITS = "en:grapefruits"
    GRAPES = "en:grapes"
    GREEN_BEANS = "en:green-beans"
    GREEN_SWEET_PEPPERS = "en:green-sweet-peppers"
    KIWIS = "en:kiwis"
    KAKIS = "en:kakis"
    LEEKS = "en:leeks"
    LEMONS = "en:lemons"
    LETTUCES = "en:lettuces"
    LIMES = "en:limes"
    LYCHEES = "en:lychees"
    MANDARIN_ORANGES = "en:mandarin-oranges"
    MANGOES = "en:mangoes"
    MELONS = "en:melons"
    MUSHROOMS = "en:mushrooms"
    NECTARINES = "en:nectarines"
    ONIONS = "en:onions"
    ORANGES = "en:oranges"
    PAPAYAS = "en:papayas"
    PARSNIP = "en:parsnip"
    PASSION_FRUITS = "en:passion-fruits"
    PEACHES = "en:peaches"
    PEARS = "en:pears"
    PEAS = "en:peas"
    PEPPERS = "en:peppers"
    PINEAPPLE = "en:pineapple"
    PLUMS = "en:plums"
    POMEGRANATES = "en:pomegranates"
    POMELOS = "en:pomelos"
    POTATOES = "en:potatoes"
    PUMPKINS = "en:pumpkins"
    RADISHES = "en:radishes"
    RASPBERRIES = "en:raspberries"
    RED_BELL_PEPPERS = "en:red-bell-peppers"
    RED_ONIONS = "en:red-onions"
    RHUBARBS = "en:rhubarbs"
    SCALLIONS = "en:scallions"
    SHALLOTS = "en:shallots"
    SPINACHS = "en:spinachs"
    SPROUTS = "en:sprouts"
    STRAWBERRIES = "en:strawberries"
    TOMATOES = "en:tomatoes"
    TURNIP = "en:turnip"
    WATERMELONS = "en:watermelons"
    WALNUTS = "en:walnuts"
    YELLOW_ONIONS = "en:yellow-onions"
    ZUCCHINI = "en:zucchini"
    OTHER = "other"


class Origin(enum.StrEnum):
    FRANCE = "en:france"
    ITALY = "en:italy"
    SPAIN = "en:spain"
    POLAND = "en:poland"
    CHINA = "en:china"
    BELGIUM = "en:belgium"
    MOROCCO = "en:morocco"
    PERU = "en:peru"
    PORTUGAL = "en:portugal"
    MEXICO = "en:mexico"
    OTHER = "other"
    UNKNOWN = "unknown"


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
    price: float = Field(..., description="Price in the local currency")
    currency: str | None = Field(
        ...,
        description="Currency of the price.",
    )
    price_per: Unit = Field(..., description="Unit of the price")
    price_is_discounted: bool = Field(
        False, description="true if the price is discounted, false otherwise"
    )
    price_without_discount: float | None = Field(
        ...,
        description="The price without discount, if the price is discounted. "
        "If the price is not discounted, this should be set to None.",
    )
    discount_type: DiscountType | None = Field(
        None,
        description="The type of discount applied to the price, if any. "
        "If no discount is applied, this should be set to None.",
    )
    with_vat: bool = Field(
        True,
        description="True if the price includes VAT (Value Added Tax), false otherwise.",
    )


class LabelPrice(BaseModel):
    """One of the price displayed on a price tag.

    For raw products (without barcode), if the price is indicated per weight
    but is not per kilogram (example: per 100g or per 500g), the price per
    kilogram should be calculated.
    """

    price: float = Field(..., description="Price in the local currency")
    with_vat: bool = Field(
        True,
        description="True if the price includes VAT (Value Added Tax), false otherwise. "
        "If there are no VAT in the country, set to false. In most cases, this should be set to true.",
    )
    currency: str | None = Field(
        ...,
        description="Currency of the price. Set to null if unknown. Examples: 'EUR', 'USD', 'GBP'",
    )
    price_per: Unit = Field(..., description="Unit of the price")
    price_is_discounted: bool = Field(
        False, description="true if the price is discounted, false otherwise"
    )
    discount_type: DiscountType = Field(
        DiscountType.NO_DISCOUNT,
        description="The type of discount applied to the price, if any. "
        "If no discount is applied, this should be set to NO_DISCOUNT.",
    )


class Label(BaseModel):
    """A label (also called price tag) indicates in a store the price of the
    product and possibly other information such as the category, price per kg,
    origin, etc.

    We distinguish between two types of labels:

    - Labels for packaged products, with barcode (type: PRODUCT): these are
    products that have a barcode, which is usually displayed on the price tag.
    For this type of label, the category should be set to NO_CATEGORY, the
    origin should be set to NO_ORIGIN.
    - Raw products, without barcode (type: CATEGORY): these are usually fruits
    and vegetables, but it can also be any product sold per weight (kg, 100g,
    etc.). For this type of label, the category should be set to the
    corresponding RawCategory, the origin should be set to the country of
    origin of the product (if indicated) and the barcode should be empty.
    """

    type: Literal["PRODUCT", "CATEGORY"] = Field(
        ...,
        description="The type of product the label is referring to. It should be "
        "`PRODUCT` for packaged products with barcode, and `CATEGORY` for raw "
        "products without barcode.",
    )
    category: RawCategory | None = Field(
        None,
        description="The category of the product. "
        "If type=PRODUCT, this should be set to the null.",
    )  # category_tag
    prices: list[LabelPrice] = Field(
        ...,
        description="All prices found on the label. Depending on the type of "
        "price tag, there can be multiple prices displayed. "
        "For packaged products (type=PRODUCT), the price per unit is the most common one. "
        "The price per kg is also often included as well. "
        "For raw products (type=CATEGORY), the price per kg is the most common one. "
        "There can also be a discount applied to the price. In such case, both "
        "the original price and the discounted price should be included in the list.",
    )
    origin: Origin | None = Field(
        ...,
        description="The country of origin of the product. "
        "If type=PRODUCT, this should be set to null. If type=CATEGORY, this should "
        "be set to the country of origin of the product, such as France, Italy, Spain, etc.",
    )
    organic: bool = Field(
        ...,
        description="true if the product is organic, false otherwise. If true, "
        "there should be evidence on the label suggesting that the product is "
        "organic, such as the EU organic logo.",
    )
    barcode: str = Field(
        ...,
        description="The barcode of the product, if available. "
        "The barcode are usually numbers with 13 (EAN13) or 8 (EAN8) digits. You should "
        "*NOT* try to decode the barcode stripe (also called modules), but use the "
        "barcode number displayed on the label. "
        "If type=CATEGORY, this should be empty.",
    )
    product_name: str = Field(
        ...,
        description="The name of the product, as displayed on the label. "
        "For raw products (type=CATEGORY), this is usually the name of the fruit or vegetable. "
        "For products with barcode (type=PRODUCT), it usually includes the brand, a short "
        "description of the product, and eventually the quantity. "
        "examples: 'NOCCIOLATA BIO 650G', 'Simpson Donuts', 'GERBLE BISCUIT PIST ABRICOT160G', "
        "'Radis Blanc', 'Concombre lisse', 'Courget Butternut', 'Tomatoes', 'Organic Bananas'",
    )
    truncated: bool = Field(
        False,
        description="true if the photo of the price tag is truncated, false otherwise. A photo of a price tag is considered truncated if any of these occurs:\n"
        "- the barcode (for packaged products) or the product name (for raw products) is not fully visible on the photo\n"
        "- the price is not fully visible on the photo\n",
    )
    is_price_tag: bool = Field(
        True,
        description="true if the image is a price tag, false otherwise. If the image seems to come from a receipt or a catalogue, this should be set to false.",
    )

    @computed_field
    @property
    def selected_price(self) -> SelectedPrice | None:
        """From all individual price reference on the price tag, construct a
        Price ready to be added to Open Prices.

        In case
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
    category_tag: str | None = None
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
