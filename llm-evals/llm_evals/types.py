from typing import Any, Callable, Literal

from pydantic import BaseModel
from pydantic_evals import Dataset

TaskType = Literal[
    "food:product_info_extraction",
    "food:product_categorization",
    "prices:price_tag_extraction",
]


OutputMode = Literal["tool", "native", "prompted"]


class TaskConfig(BaseModel):
    dataset: Dataset
    task: Callable[[dict[str, Any]], Any]
    instructions: str
    multiple_images: bool
    output_type: type[BaseModel]
