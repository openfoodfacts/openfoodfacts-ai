import logging
from pathlib import Path
import os
import json

import pandas as pd

from spellcheck import SpellChecker
from utils.llm import OpenAIChatCompletion
from utils.prompt import SystemPrompt, Prompt
from utils.model import OpenAIModel


REPO_DIR = Path(os.path.dirname(__file__)).parent.parent
DATA_PATH = REPO_DIR / "data/dataset/extracted_lists_of_ingredients.parquet"
SYNTHETIC_DATA_PATH = REPO_DIR / "data/dataset/synthetic_data.jsonl"

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    df = pd.read_parquet(DATA_PATH)

    existing_codes = (
        pd.read_json(SYNTHETIC_DATA_PATH, lines=True)["code"].to_list()
        if SYNTHETIC_DATA_PATH.exists()
        else []
    )

    spellchecker = SpellChecker(
        model=OpenAIModel(
            llm=OpenAIChatCompletion(
                prompt_template=Prompt.spellcheck_prompt_template,
                system_prompt=SystemPrompt.spellcheck_system_prompt,
            )
        )
    )
    with open(SYNTHETIC_DATA_PATH, "a") as file:
        for _, row in df.iterrows():
            if row["code"] in existing_codes:
                LOGGER.info("Product was already generated. Pass")
            else:
                row["corrected_text"] = spellchecker.predict(row["ingredients_text"])
                json.dump(row.to_dict(), file, ensure_ascii=False) # For accents
                file.write("\n")
                file.flush() # Immediatly write the line into the file
    LOGGER.info("Synthetic genration finished.")


def generate_synthetic_data(
    df: pd.DataFrame,
    spellchecker: SpellChecker,
    feature_name: str,
    output_path: Path,
    synthetic_feature_name: str = "correction_text",
) -> pd.DataFrame:
    """"""
    LOGGER.info("Start generating synthetic data.")


if __name__ == "__main__":
    main()
