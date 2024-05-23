import json
from pathlib import Path
from typing import List

import pandas as pd

from spellcheck.spellcheck import Spellcheck
from spellcheck.prompt import SystemPrompt, Prompt
from spellcheck.model import OpenAIChatCompletion
from spellcheck.utils import get_logger, get_repo_dir


REPO_DIR = get_repo_dir()
DATA_PATH = REPO_DIR / "data/dataset/0_extracted_lists_of_ingredients.parquet"
SYNTHETIC_DATA_PATH = REPO_DIR / "data/dataset/1_synthetic_data_1.jsonl"

MODEL_NAME = "gpt-3.5-turbo"

LOGGER = get_logger()


def main():
    
    df = pd.read_parquet(DATA_PATH)
    existing_codes = (
        pd.read_json(SYNTHETIC_DATA_PATH, lines=True)["code"].to_list()
        if SYNTHETIC_DATA_PATH.exists()
        else []
    )
    spellcheck = Spellcheck(
        model=OpenAIChatCompletion(
            prompt_template=Prompt.spellcheck_prompt_template,
            system_prompt=SystemPrompt.spellcheck_system_prompt,
            model_name=MODEL_NAME
        )
    )
    generate_synthetic_data(
        df=df,
        output_data_path=SYNTHETIC_DATA_PATH,
        existing_codes=existing_codes,
        spellcheck=spellcheck,
        original_text_feature="ingredients_text",
        synthetic_feature = 'corrected_text'
    )


def generate_synthetic_data(
    df: pd.DataFrame,
    output_data_path: Path,
    existing_codes: List,
    spellcheck: Spellcheck,
    original_text_feature: str,
    synthetic_feature: str
) -> None:
    """Generate synthetic data for text-based features using a spellcheck.

    Notes:
    - This function appends synthetic data to an existing file specified by output_data_path.
    - Each row in the DataFrame is processed, and if the code is not in the list of existing codes, the text in the original_text_feature column is corrected using the spellcheck and appended to the output file along with other data.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the original data.
    - output_data_path (Path): Path to save the synthetic data.
    - existing_codes (List): List of existing codes to skip already generated products.
    - spellcheck (Spellcheck): Spellchecker object to correct text features.
    - original_text_feature (str): Name of the column containing original text data.
    - synthetic_feature (str): Name of the column to store synthetic data.


    """
    with open(output_data_path, "a") as file:
        for _, row in df.iterrows():
            if row["code"] in existing_codes:
                LOGGER.info("Product was already generated. Pass")
            else:
                row[synthetic_feature] = spellcheck.correct(row[original_text_feature])
                json.dump(row.to_dict(), file, ensure_ascii=False) # Ensure ascii for accents
                file.write("\n")
                file.flush() # Immediatly write the line into the file
    LOGGER.info("Synthetic generation finished.")


if __name__ == "__main__":
    main()
