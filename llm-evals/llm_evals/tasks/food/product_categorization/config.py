from llm_evals.types import TaskConfig

from .datasets import dataset
from .schemas import CategoryPredictionResponseModel

DEFAULT_INSTRUCTIONS = "Predict the categories of this product."


CONFIG = TaskConfig(
    dataset=dataset,
    instructions=DEFAULT_INSTRUCTIONS,
    output_type=CategoryPredictionResponseModel,
    name="product_categorization",
)
