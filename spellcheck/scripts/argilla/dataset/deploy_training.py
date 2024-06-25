from dotenv import load_dotenv

from spellcheck.utils import get_repo_dir
from spellcheck.argilla_modules import TrainingDataArgilla
from spellcheck.utils import get_logger

LOGGER = get_logger("INFO")

load_dotenv()

REPO_DIR = get_repo_dir()
SYNTHETIC_DATASET = REPO_DIR / "data/dataset/1_synthetic_data.jsonl"

ARGILLA_DATASET_NAME = "training_dataset_v2"
ARGILLA_WORKSPACE_NAME = "spellcheck"


if __name__ == "__main__":
    
    TrainingDataArgilla.from_dataset(
        hf_repo="openfoodfacts/spellcheck-dataset",
        split="train+test",
        original_feature="text",
        reference_feature="label",
    ).deploy(
        dataset_name=ARGILLA_DATASET_NAME, 
        workspace_name=ARGILLA_WORKSPACE_NAME,
    )
    