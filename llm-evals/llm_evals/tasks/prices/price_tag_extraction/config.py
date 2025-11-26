from llm_evals.types import TaskConfig

from .datasets import dataset
from .evaluate import agent, task

CONFIG = TaskConfig(
    agent=agent,
    dataset=dataset,
    task=task,
)
