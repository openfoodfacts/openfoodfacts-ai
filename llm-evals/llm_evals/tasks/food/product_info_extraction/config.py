from llm_evals.types import TaskConfig

from .datasets import dataset
from .tasks import agent, task

CONFIG = TaskConfig(
    agent=agent,
    dataset=dataset,
    task=task,
)
