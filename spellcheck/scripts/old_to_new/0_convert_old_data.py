"""Extract the old data prepared by Lucain W and save it as a json file for future work.
"""
import os
from pathlib import Path
from typing import Iterator, Mapping, List
import json


SPELLCHECK_DIR = Path(os.path.dirname(__file__)).parent.parent
OLD_DATA_DIR = SPELLCHECK_DIR / "old/test_sets/fr/uniform_sampling"
ORIGINAL_DATA_PATH = OLD_DATA_DIR / "original.txt"
REFERENCE_DATA_PATH = OLD_DATA_DIR / "correct.txt"
OUTPUT_DATA_PATH = SPELLCHECK_DIR / "data/0_fr_data.json"


def convert_old_data(original_path: Path, reference_path: Path, save_path: Path):
    """Convert data from previous work from Lucain W into Json format.

    Args:
        original_path (Path): Ingredient Lists before Spellcheck
        reference_path (Path): Ingredient Lists after Spellcheck
        save_path (Path): save path
    """
    original_texts = extract_texts_from_data(original_path)
    reference_texts = extract_texts_from_data(reference_path)
    new_data = postprocess_texts(original_texts, reference_texts)
    save_to_json(new_data, save_path)


def extract_texts_from_data(path: Path) -> Iterator[str]:
    """Data is stored as .txt file, each file containing the barcode and the list of ingredients.
    Check it at `spellcheck/old/test_sets/fr/uniform_sampling`

    Args:
        path (Path): .txt filepath containinng the list of Ingredients + barcode

    Yields:
        Iterator[str]: List of Ingredients as text
    """
    with open(path, "r") as f:
        for line in f.readlines():
            yield line.split("\t")[1].strip("\n")


def postprocess_texts(
    original_texts: Iterator[str],
    reference_texts: Iterator[str],
    unaccepted_string: str = "NOT_VALID",
    lang: str = "fr"
) -> List[Mapping]:
    """Map original and reference texts. Remove 

    Args:
        original_texts (Iterator[str]): Before Spellchecking
        reference_texts (Iterator[str]): After Spellchecking
        unaccepted_string (str, optional): Some `After spellcheck` data were considered as not valid. 
    We remove them. Defaults to "NOT_VALID".
        lang (str, optional): Langue. Defaults to "fr".

    Returns:
        List[Mapping]: List of original and reference list of ingredients
    """
    output = []
    for original_text, reference_text in zip(original_texts, reference_texts):
        if reference_text != unaccepted_string:
            output.append(
                {"original": original_text, "reference": reference_text, "lang": lang}
            )
    return output


def save_to_json(data: Iterator[Mapping], path: Path) -> None:
    """Saving.
    Args:
        data (Iterator[Mapping]): List of original and reference list of ingredients
        path (Path): Saving path
    """
    obj = {"data": data}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    convert_old_data(
        original_path=ORIGINAL_DATA_PATH,
        reference_path=REFERENCE_DATA_PATH,
        save_path=OUTPUT_DATA_PATH
    )


if __name__ == "__main__":
    main()
