"""Using the data we manually labeled, let's test our benchmark building algorithm on it."""

from typing import Iterator, Mapping
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


def load_labeled_data(path: Path) -> Iterator[Iterator[str]]:
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

    data = load_labeled_data(path=LABELED_DATA_PATH)

    spellchecker = SpellChecker(
        model=OpenAIModel(
            llm=OpenAIChatCompletion(
                prompt_template=Prompt.spellcheck_prompt_template,
                system_prompt=SystemPrompt.spellcheck_system_prompt,
            )
        )
    )

    output_data = [
        {
            "original": original,
            "reference": reference,
            "prediction": spellchecker.predict(original)
        } for original, reference, _ in data
    ]

    save_data(
        data={"data": output_data},
        save_path=BENCHMARK_PATH
    )

    LOGGER.info("Over.")