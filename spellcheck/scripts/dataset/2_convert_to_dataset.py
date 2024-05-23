import json

from datasets import Dataset

from spellcheck.utils import get_repo_dir


REPO_DIR = get_repo_dir()
DATA_PATH = REPO_DIR / "data/dataset/1_synthetic_data.jsonl"

HF_REPO = "openfoodfacts/spellcheck-dataset"
COMMIT_MESSAGE = "Dataset-v2"


def main():
    with open(DATA_PATH, "r") as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]

    dataset = (
        Dataset.from_list(data)
        .remove_columns(["ingredients_n", "unknown_ingredients_n", "fraction"])
        .rename_columns({"ingredients_text": "text", "corrected_text": "label"})
    )
    dataset.push_to_hub(
        repo_id=HF_REPO,
        commit_message=COMMIT_MESSAGE
    )


if __name__ == "__main__":
    main()