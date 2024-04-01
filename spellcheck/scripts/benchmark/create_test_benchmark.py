"""Using the data we manually labeled, let's test our benchmark building algorithm on it."""

from typing import Iterable, Mapping, List
from pathlib import Path
import os
import logging
import json

from spellcheck import SpellChecker
from utils.llm import OpenAIChatCompletion
from utils.prompt import SystemPrompt, Prompt
from utils.model import OpenAIModel


SPELLCHECK_DIR = Path(os.path.dirname(__file__)).parent.parent
LABELED_DATA_PATH = SPELLCHECK_DIR / "data/labeled/corrected_list_of_ingredients.txt"
BENCHMARK_PATH = SPELLCHECK_DIR / "data/benchmark/test_benchmark.json"

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def prepare_test_benchmark(
    labeled_data_path: Path, 
    spellchecker: SpellChecker, 
    save_path: Path
) -> None:
    """Preparation of the test benchmark using labeled data. 
    This step helps us prompt engineering GPT-3.5/GPT-4 to later augmentte our data.

    Args:
        labeled_data_path (Path): Manually labeled list of ingredients
        spellchecker (SpellChecker): Spellcheck module
        save_path (Path): Test benchmark save path
    """
    data = load_labeled_data(labeled_data_path)
    output_data = {"data": [
            {
                "original": original,
                "reference": reference,
                "openai_prediction": spellchecker.predict(original)
            } for original, reference, _ in data
        ]
    }
    save_data(
        data=output_data,
        save_path=save_path
    )
    LOGGER.info("Test benchmark created and saved.")


def load_labeled_data(path: Path) -> Iterable[List[str]]:
    """"""
    with open(path, "r") as f:
        data = f.read()
    elements = [
        element.split("\n") for element in data.split("\n\n")
    ]  # Shape (N, 3): [[original, reference, lang], ...]
    LOGGER.info(f"First element of the labeled data: {elements[0]}")
    return elements


def save_data(data: Mapping, save_path: Path) -> None:
    """"""
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    spellchecker = SpellChecker(
        model=OpenAIModel(
            llm=OpenAIChatCompletion(
                prompt_template=Prompt.spellcheck_prompt_template,
                system_prompt=SystemPrompt.spellcheck_system_prompt
            )
        )
    )
    prepare_test_benchmark(
        labeled_data_path=LABELED_DATA_PATH,
        spellchecker=spellchecker,
        save_path=BENCHMARK_PATH
    )