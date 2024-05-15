"""Spellcheck models"""
from abc import ABC, abstractmethod
from typing import Literal

from openai import OpenAI
from anthropic import Client


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
        model_name: Literal["gpt-3.5-turbo", "gpt-4-turbo"],
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
        messages = self.messages + [{"role": "user", "content": self.prompt_template.format(text)}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        output_text = response.choices[0].message.content
        return output_text.strip()


class AnthropicChatCompletion(BaseModel):
    """LLMs from Anthropic
    """
    def __init__(
        self,
        prompt_template: str,
        system_prompt: str,
        model_name: Literal[
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229"
        ],
        temperature: float = 0,
        max_tokens: int = 512,
    ) -> None:
        self.client = Client()
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, text: str) -> str:
        message = self.client.messages.create(
            model=self.model_name,
            system=self.system_prompt,
            messages=[{"role": "user", "content": self.prompt_template.format(text)}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return message.content[0].text
    

class RulesBasedModel(BaseModel):
    """Rules-based methods."""

    @staticmethod
    def generate(text: str) -> str:
        return text.replace("léci - thine", "lécithine")