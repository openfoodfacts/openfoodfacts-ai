from typing import Any, Callable, Literal, TypedDict

from pydantic_ai import Agent
from pydantic_evals import Dataset

TaskType = Literal["food:product_info_extraction", "prices:price_tag_extraction"]


class TaskConfig(TypedDict):
    agent: Agent[None, Any]
    dataset: Dataset
    task: Callable[[dict[str, Any]], Any]
