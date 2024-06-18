"""Append new ingredients lists corrections using LLM to the benchmark."""
import pandas as pd

from spellcheck.spellcheck import Spellcheck
from spellcheck.model import OpenAIChatCompletion
from spellcheck.prompt import SystemPrompt, Prompt
from spellcheck.utils import get_repo_dir, get_logger


LOGGER = get_logger()

REPO_DIR = get_repo_dir()
ADDITIONAL_PRODUCTS_PATH = REPO_DIR / "data/benchmark/additional_products/extracted_additional_products.parquet"
SYNTHETIC_DATA_PATH = REPO_DIR / "data/benchmark/additional_products/synthetically_corrected_products.parquet"

MODEL_NAME = "gpt-3.5-turbo"


def main():

    # Init
    spellcheck = Spellcheck(
        model=OpenAIChatCompletion(
            prompt_template=Prompt.spellcheck_prompt_template,
            system_prompt=SystemPrompt.spellcheck_system_prompt,
            model_name=MODEL_NAME
        )
    )

    # Process
    df = pd.read_parquet(ADDITIONAL_PRODUCTS_PATH)
    LOGGER.info(f"Features: {df.columns}")
    df["correction"] = df["ingredients_text"].apply(spellcheck.correct)

    # Save
    df.to_parquet(SYNTHETIC_DATA_PATH)
    
if __name__ == "__main__":
    main()