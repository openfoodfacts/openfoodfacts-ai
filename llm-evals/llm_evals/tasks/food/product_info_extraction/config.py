from llm_evals.types import TaskConfig

from .datasets import dataset
from .schemas import ProductInfoExtractionResponseModel
from .tasks import DEFAULT_INSTRUCTIONS, task

CONFIG = TaskConfig(
    dataset=dataset,
    task=task,
    instructions=DEFAULT_INSTRUCTIONS,
    multiple_images=False,
    output_type=ProductInfoExtractionResponseModel,
    name="food_product_info_extraction",
)
