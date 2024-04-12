"""Spellcheck models"""
from abc import ABC, abstractmethod

from utils.llm import LLM


class BaseModel(ABC):
    """Base model used by the Spellcheck."""
    @abstractmethod
    def predict(self, text: str) -> str:
        raise NotImplementedError


class OpenAIModel(BaseModel):
    """OpenAI module.
    
    Args:
        llm (LLM): OpenAI LLM
    """
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def predict(self, text: str) -> str:
        """Text generation from OpenAI

        Args:
            text (str): Input

        Returns:
            str: Generated text.
        """
        prompt = self.llm.prompt_template.format(text)
        output = self.llm.generate(prompt=prompt)
        return output