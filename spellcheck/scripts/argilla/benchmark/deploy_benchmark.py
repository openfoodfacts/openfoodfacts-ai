from spellcheck.utils import get_repo_dir
from spellcheck.argilla_modules import BenchmarkArgilla

REPO_DIR = get_repo_dir()
BENCHMARK_PATH = REPO_DIR / "data/benchmark/verified_benchmark.parquet"

ARGILLA_DATASET_NAME = "benchmark_v4"
ARGILLA_WORKSPACE_NAME = "spellcheck"


if __name__ == "__main__":

    BenchmarkArgilla.from_parquet(path=BENCHMARK_PATH).deploy(
        dataset_name=ARGILLA_DATASET_NAME, 
        workspace_name=ARGILLA_WORKSPACE_NAME
    )
