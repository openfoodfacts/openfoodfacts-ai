"""Spellcheck models"""
from abc import ABC, abstractmethod
from typing import Literal

from openai import OpenAI


class BaseModel(ABC):
    """Base model used by the Spellcheck."""
    @abstractmethod
    def generate(self, text: str) -> str:
        raise NotImplementedError


class OpenAIChatCompletion(BaseModel):
    """ChatGPT from OpenAI API

    Init:
        prompt_template (str): _description_
        system_prompt (str): _description_
        model_name (Literal["gpt-3.5-turbo"], optional): _description_. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): _description_. Defaults to 0.
        max_tokens (int, optional): _description_. Defaults to 512.
    """
    def __init__(
        self,
        prompt_template: str,
        system_prompt: str,
        model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo"] = "gpt-3.5-turbo",
        temperature: float = 0,
        max_tokens: int = 512,
    ) -> None:
        self.client = OpenAI()
        self.messages = [{"role": "system", "content": system_prompt}]
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, text: str) -> str:
        """Generate assistant response.

        Args:
            prompt (str): Instruction prompt

        Returns:
            str: text completion
        """
        messages = self.messages + [{"role": "user", "content": self.prompt_template.format(text)}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        output_text = response.choices[0].message.content
        return output_text.strip()

