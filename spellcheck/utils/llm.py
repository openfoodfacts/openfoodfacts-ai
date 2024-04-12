import logging
from abc import ABC, abstractmethod

from openai import OpenAI


LOGGER = logging.getLogger(__name__)


class LLM(ABC):
    """OpenAI LLMs for text generation.

    Args:
    prompt_template (str): Prompt template
    model_name (str): Model name from OpenAI: https://platform.openai.com/docs/models
    temperature (float): Temperature coefficient
    max_tokens (int): Max tokens output
    """
    def __init__(
        self, 
        prompt_template: str, 
        model_name: str, 
        temperature: float, 
        max_tokens: int
    ) -> None:
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIChatCompletion(LLM):
    "ChatGPT from OpenAI API"

    def __init__(
        self,
        prompt_template: str,
        system_prompt: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        max_tokens: int = 512,
    ) -> None:
        self.client = OpenAI()
        self.messages = [{"role": "system", "content": system_prompt}]
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Generate assistant response.

        Args:
            prompt (str): Instruction prompt

        Returns:
            str: text completion
        """
        messages = self.messages + [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        output_text = response.choices[0].message.content
        return output_text.strip()


if __name__ == "__main__":
    prompt = "Hello, how are you?"
    chat = OpenAIChatCompletion(
        system_prompt="You're a friendly bot assistant passionate about basketball."
    )
    output = chat.generate(prompt)
    print(output)
