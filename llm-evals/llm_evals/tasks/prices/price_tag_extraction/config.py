from llm_evals.types import TaskConfig

from .datasets import dataset
from .schemas import Label

DEFAULT_INSTRUCTIONS = (
    "Here is one picture containing a price label, extract information "
    "from it. If you cannot decode an attribute, set it to an empty string."
)

CONFIG = TaskConfig(
    dataset=dataset,
    instructions=DEFAULT_INSTRUCTIONS,
    output_type=Label,
    name="price_price_tag_extraction",
)
