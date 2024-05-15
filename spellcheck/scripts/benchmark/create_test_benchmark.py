"""Using the data we manually labeled, let's test our benchmark building algorithm on it."""

from typing import Iterable, Mapping, List
from pathlib import Path
import json

from spellcheck.spellcheck import Spellcheck
from spellcheck.prompt import SystemPrompt, Prompt
from spellcheck.model import OpenAIChatCompletion
from spellcheck.utils import get_logger, get_repo_dir


LOGGER = get_logger()

REPO_DIR = get_repo_dir()
LABELED_DATA_PATH = REPO_DIR / "data/labeled/corrected_list_of_ingredients.txt"
BENCHMARK_PATH = REPO_DIR / "data/benchmark/test_benchmark.json"

MODEL_NAME = "gpt-3.5-turbo"


def main():
    spellcheck = Spellcheck(
        model=OpenAIChatCompletion(
            prompt_template=Prompt.spellcheck_prompt_template,
            system_prompt=SystemPrompt.spellcheck_system_prompt,
            model_name=MODEL_NAME
        )
    )
    prepare_test_benchmark(
        labeled_data_path=LABELED_DATA_PATH,
        spellcheck=spellcheck,
        save_path=BENCHMARK_PATH
    )


def prepare_test_benchmark(
    labeled_data_path: Path, 
    spellcheck: Spellcheck, 
    save_path: Path
) -> None:
    """Preparation of the test benchmark using labeled data. 
    This step helps us prompt engineering GPT-3.5/GPT-4 to later augmentte our data.

    Args:
        labeled_data_path (Path): Manually labeled list of ingredients
        spellcheck (Spellcheck): Spellcheck module
        save_path (Path): Test benchmark save path
    """
    data = load_labeled_data(labeled_data_path)
    output_data = {"data": [
            {
                "original": original,
                "reference": reference,
                "openai_prediction": spellcheck.correct(original)
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
    main()