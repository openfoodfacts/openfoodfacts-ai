from llm_evals.tasks.prices.receipt_anonymization.schemas import PersonalInfoList
from llm_evals.types import TaskConfig

from .datasets import CUSTOM_EVALUATOR_TYPES, DATASET_PATH, dataset

DEFAULT_INSTRUCTIONS = """Identify the following personal information from this receipt:
- name of the supermarket cashier, if any. It should only be included in the results if the first name and/or last name of the cashier is mentioned.
- name of the buyer (who may have used a fidelity card). Some street names may contain name of people (as the address of the shop is often displayed on the receipt), but they should not be included in the results.
- fidelity card ID
- hour of the purchase (if available). Only the hour is needed, not the date.

For the value:
- don't change the formatting, keep the value as it is displayed on the receipt (e.g. if the hour is displayed as '14h30', keep it as is, don't convert it to '14:30').
- only include the personal information, not the suffix (ex: don't include 'fidelity card ID:', but only the ID).

If no personal information was found, return an empty list."""

CONFIG = TaskConfig(
    dataset=dataset,
    dataset_path=DATASET_PATH,
    custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    instructions=DEFAULT_INSTRUCTIONS,
    output_type=PersonalInfoList,
    name="price_receipt_anonymization",
)
