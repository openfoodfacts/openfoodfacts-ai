from typing import Any, Callable, Literal

from pydantic import BaseModel
from pydantic_evals import Dataset

TaskType = Literal[
    "food:product_info_extraction",
    "food:product_categorization",
    "prices:price_tag_extraction",
]


OutputMode = Literal["tool", "native", "prompted", "native+prompted"]


class TaskConfig(BaseModel):
    dataset: Dataset
    task: Callable[[dict[str, Any]], Any] | None = None
    instructions: str
    output_type: type[BaseModel]
    name: str
    add_sample_func: Callable[[str], None] | None = None
