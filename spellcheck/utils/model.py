"""Spellcheck models"""
from abc import ABC, abstractmethod

from utils.llm import LLM


class BaseModel(ABC):
    
    @abstractmethod
    def predict(self, text: str) -> str:
        pass


class OpenAIModel(BaseModel):
    """Spellcheck based on OpenAI LLMs"""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def predict(self, text: str) -> str:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        prompt = self.llm.prompt_template.format(text)
        output = self.llm.generate(prompt=prompt)
        return output