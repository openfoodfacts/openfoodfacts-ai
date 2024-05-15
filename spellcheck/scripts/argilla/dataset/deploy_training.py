from dotenv import load_dotenv

from spellcheck.utils import get_repo_dir
from spellcheck.argilla_modules import TrainingDataArgilla


load_dotenv()

REPO_DIR = get_repo_dir()
SYNTHETIC_DATASET = REPO_DIR / "data/dataset/1_synthetic_data.jsonl"

ARGILLA_DATASET_NAME = "training_dataset"
ARGILLA_WORKSPACE_NAME = "spellcheck"


if __name__ == "__main__":  
    
    TrainingDataArgilla.from_jsonl(path=SYNTHETIC_DATASET).deploy(
        dataset_name=ARGILLA_DATASET_NAME, workspace_name=ARGILLA_WORKSPACE_NAME
    )
    