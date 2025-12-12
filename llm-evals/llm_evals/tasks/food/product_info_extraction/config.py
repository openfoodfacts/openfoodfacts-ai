from llm_evals.types import TaskConfig

from .datasets import dataset
from .schemas import ProductInfoExtractionResponseModel

DEFAULT_INSTRUCTIONS = (
    "Extract all relevant information from this product packaging photo."
)

CONFIG = TaskConfig(
    dataset=dataset,
    instructions=DEFAULT_INSTRUCTIONS,
    output_type=ProductInfoExtractionResponseModel,
    name="food_product_info_extraction",
)
