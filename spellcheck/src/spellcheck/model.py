"""Spellcheck models"""
import os
from abc import ABC, abstractmethod
from typing import Literal
from dotenv import load_dotenv

from openai import OpenAI
from anthropic import Client
import vertexai
from vertexai.generative_models import GenerativeModel


load_dotenv()


class BaseModel(ABC):
    """Base model used by the Spellcheck."""

    @abstractmethod
    def generate(self, text: str) -> str:
        """From a string, return a response."""
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
    

class GeminiModel(BaseModel):
    """Google Gemini."""

    def __init__(
        self,
        system_prompt: str,
        prompt_template: str,
        model_name: Literal[
            "gemini-1.0-pro-002",
            "gemini-1.5-flash-preview-0514",
            "gemini-1.5-pro-preview-0409" 
        ], 
        temperature: float = 0,
        max_tokens: int= 512,
        project_id: str = "robotoff",
        location: str = "us-central1"
    ) -> None:
        self.prompt_template = prompt_template

        # Init model 
        vertexai.init(project=project_id, location=location)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain"
        }
        self.model = GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_prompt,
        )

    def generate(self, text: str) -> str:
        response = self.model.generate_content(
            self.prompt_template.format(text),
        )
        finish_reason = response.to_dict()["candidates"][0]["finish_reason"]
        # OK
        if finish_reason == "STOP":
            return response.text
        # Not OK such as "RECITATION", which means Google found out the text is copied from a web page
        else:
            return text


class LLMInferenceEndpoint(BaseModel):
    """Open-Source LLM deployed on Hugging Face Inference Endpoints."""

    def __init__(
            self, 
            prompt_template: str, 
            system_prompt: str,
            temperature: int = 0, 
            max_tokens: int = 512
        ) -> None:
        self.client = OpenAI(
            base_url=os.getenv("HF_INFERENCE_ENDPOINT_URL") + "/v1/",
            api_key=os.getenv("HF_TOKEN")
        ) # With TGI, we can use the existing OpenAI API to run our LLM
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, text: str) -> str:
        message = self.client.chat.completions.create(
            model="tgi",
            messages=[
                {"role": "user", "content": self.system_prompt + "\n\n" + self.prompt_template.format(text)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return message.choices[0].message.content


