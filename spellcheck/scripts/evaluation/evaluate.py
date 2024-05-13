"""Using the existing benchmark and evaluation algorithm, evaluate exisiting LLM APIs
* OpenAI GPT-3.5, GPT-4
* Gemini
* Claude 3
* ...
"""

from typing import Iterable, Mapping, Tuple	
import json
from pathlib import Path
from dotenv import load_dotenv
import time
from datetime import datetime
from tqdm import tqdm

import pandas as pd

from utils.utils import get_logger, get_repo_dir
from utils.evaluation import SpellcheckEvaluator
from spellcheck import Spellcheck
from utils.model import AnthropicChatCompletion, OpenAIChatCompletion, RulesBasedModel
from utils.prompt import SystemPrompt, Prompt
from utils.argilla_modules import BenchmarkEvaluationArgilla, IngredientsCompleteEvaluationArgilla


REPO_DIR = get_repo_dir()
BENCHMARK_PATH = REPO_DIR / "data/benchmark/verified_benchmark.parquet"
INGREDIENTS_COMPLETE_DATA_PATH = REPO_DIR / "data/database/ingredients_complete.parquet"

# Metrics
METRICS_PATH = REPO_DIR / "data/evaluation/metrics.jsonl"

MODEL_NAME = "gpt-3.5-turbo"
BENCHMARK_VERSION = "v5"
PROMPT_VERSION = "v6"
INGREDIENTS_COMPLETE_VERSION = "v1"

# Predictions JSONL paths to study the results
PREDICTIONS_EVALUATION_PATH = REPO_DIR / "data/evaluation/" / (
    MODEL_NAME 
    + "-benchmark-" + BENCHMARK_VERSION 
    + "-prompt-" + PROMPT_VERSION 
    + ".jsonl"
)
PREDICTION_INGREDIENTS_COMPLETE_PATH = REPO_DIR / "data/evaluation" / (
    MODEL_NAME
    + "-ingredients-complete-data-" + INGREDIENTS_COMPLETE_VERSION
    + "-prompt-" + PROMPT_VERSION
    + ".jsonl"
)

START = 0 # To restart the run
WAIT = 0

# Replace for gpt3.5 => "." not accepted by Argilla
ARGILLA_BENCHMARK_DATASET_NAME = f"Evaluation-{MODEL_NAME}-benchmark-{BENCHMARK_VERSION}-prompt-{PROMPT_VERSION}".replace(".", "") 
ARGILLA_INGREDIENTS_COMPLETE_DATASET_NAME = f"Evaluation-{MODEL_NAME}-ingredients-complete-{BENCHMARK_VERSION}-prompt-{PROMPT_VERSION}".replace(".", "")

LOGGER = get_logger()

load_dotenv()


def main():
    spellcheck=Spellcheck(
        model=OpenAIChatCompletion(
            prompt_template=Prompt.spellcheck_prompt_template, #If Claude, use custom prompt template
            system_prompt=SystemPrompt.spellcheck_system_prompt,
            model_name=MODEL_NAME
        )
    )

    ####################### Evaluate on benchmark
    originals, references, metadata = import_benchmark(
        benchmark_path=BENCHMARK_PATH,
        start_from=START
    )
    evaluation = Evaluate(
        model_name=MODEL_NAME,
        metrics_path=METRICS_PATH,
        benchmark_version=BENCHMARK_VERSION,
        prompt_version=PROMPT_VERSION,
        predictions_path=PREDICTIONS_EVALUATION_PATH,
    )
    evaluation.run_evaluation(
        originals=originals,
        references=references,
        metadata=metadata,
        spellcheck=spellcheck,
        wait=WAIT
    )
    evaluation.compute_metrics(
        model_name=MODEL_NAME
    )
    # Human evaluation
    BenchmarkEvaluationArgilla.from_jsonl(
        path=PREDICTIONS_EVALUATION_PATH
    ).deploy(
        dataset_name=ARGILLA_INGREDIENTS_COMPLETE_DATASET_NAME)
    

    ####################### Evaluate on Ingredient complete dataset
    originals, references, metadata = import_ingredients_complete(path=INGREDIENTS_COMPLETE_DATA_PATH)
    evaluation = Evaluate(
        model_name=MODEL_NAME,
        metrics_path=METRICS_PATH,
        benchmark_version=INGREDIENTS_COMPLETE_VERSION,
        prompt_version=PROMPT_VERSION,
        predictions_path=PREDICTION_INGREDIENTS_COMPLETE_PATH
    )
    evaluation.run_evaluation(
        originals=originals,
        references=references,
        spellcheck=spellcheck,
        metadata=metadata,
        wait=WAIT
    )
    IngredientsCompleteEvaluationArgilla.from_jsonl(path=PREDICTION_INGREDIENTS_COMPLETE_PATH).deploy(dataset_name=ARGILLA_INGREDIENTS_COMPLETE_DATASET_NAME)


class Evaluate:
    """Evaluation module to compute the performance of the Spellcheck against the benchmark.

    Args:
        model_name: Model used in the Spellcheck module.
        metrics_path (Path): Path where to append the Evaluator metrics as a Mapping object.
            This file contains all the metrics from previous runs.
        benchmark_version (str): Version of the benchmark.
        predictions_path (Path): Path where all predictions against the benchmark are stored for further analysis.
    """
    def __init__(
        self,
        model_name: str,
        metrics_path: Path,
        benchmark_version: str,
        prompt_version: str,
        predictions_path: Path,
    ) -> None:
        self.model_name = model_name
        self.metrics_path = metrics_path
        self.benchmark_version = benchmark_version
        self.prompt_version = prompt_version
        self.predictions_path = predictions_path
        
    def run_evaluation(
        self,
        originals: Iterable[str],
        references: Iterable[str],
        spellcheck: Spellcheck,
        metadata: Iterable[Mapping],
        wait: int = None
    ) -> None:
        """Run the Spellcheck module against the benchmark and store the predictions in predictions_path as a JSONL.
        Addding predictions in a JSONL file prevents API request failures to erase the processed data. 

        Args:
            originals (Iterable[str]): ists of ingredients as seen on the website.
            references (Iterable[str]): Benchmark references
            spellcheck (Spellcheck): Spellcheck module
            predictions_path (Path): Predictions saving path for further analysis
            metadata (Iterable[Mapping]): Additional metadata to save along the predictions. Defaults to None.
            wait (int, optional): Waiting time in case of number of requests per minute limited. Defaults to None.
        """
        LOGGER.info(f"Appending {str(self.predictions_path)} file.")
        with open(self.predictions_path, "a") as file:
            for original, reference, md in tqdm(
                    zip(originals, references, metadata),
                    desc="Evaluation against benchmark",
                    total=len(originals)
                ):
                timestamp = time.time()
                prediction = spellcheck.correct(original)
                md["latency"] = time.time() - timestamp
                output = {
                    "original": original,
                    "reference": reference,
                    "prediction": prediction,
                    "metadata": md
                }
                json.dump(output, file, ensure_ascii=False) # Ensure ascii for accents
                file.write("\n")
                file.flush() # Immediatly write the line into the file
                # In case Requests Per Minute are limited 
                if wait:
                    time.sleep(wait)

    def compute_metrics(self) -> None:
        """From the predictions JSONL containing the Spellcheck predictions, compute the metrics using the evaluation module. 
        """
        with open(self.predictions_path, "r") as file:
            lines = file.readlines()
            elements = [json.loads(line) for line in lines]
        originals = [element["original"] for element in elements]
        references = [element["reference"] for element in elements]
        predictions = [element["prediction"] for element in elements]
        originals, references, predictions = self.normalize(originals, references, predictions)
        evaluator = SpellcheckEvaluator(originals=originals) #TODO Remove the module call from the function 
        metrics = evaluator.evaluate(predictions, references)
        metrics_output = {
            "metrics": metrics,
            "model": self.model_name,
            "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "benchmark_version": self.benchmark_version,
            "prompt_version": self.prompt_version,
            "benchmark_size": len(predictions)
        }
        with open(self.metrics_path, "a") as file:
            json.dump(metrics_output, file, indent=4)
            file.write("\n")

    @staticmethod
    def normalize(*text_batches) -> Tuple:
        """Normalize texts to not consider some corrections during the metrics calculation.

        Args:
            Batches of texts
        Returns:
            (Tuple) Processed texts
        """
        def process(text: str) -> str:
            text = text.lower()                                           # Lowercase
            text = " ".join([token.strip() for token in text.split()])    # Normalize whitespaces
            return text
        return ([process(text) for text in text_batch] for text_batch in text_batches)


def import_benchmark(
    path: Path, 
    start_from: int = 0
) -> Tuple[Iterable[str], Iterable[str], Iterable[Mapping]]:
    """Load benchmark.
    It is possible a previous evaluation didn't go through the entire benchmark for many reasons. 
    In this case, the evaluation is restarted from a specific index instead of starting from the beginning. 

    Args:
        path (Path): Benchmark as a parquet file
        start_from (int): Row index to load the dataset from.
    Returns:
        Tuple[Iterable[str], Iterable[str], Iterable[Mapping]]: Text and Metadata from the benchmark
    """
    if path.suffix != "parquet":
        raise ValueError(f"Wrong file format. Parquet required. Instead {path.suffix} provided")
    # Benchmark
    df = pd.read_parquet(path)
    # In case the begininning of the benchmark was already processed
    df = df.iloc[start_from:]
    LOGGER.info(f"Data features: {df.columns}")
    LOGGER.info(f"Data length: {len(df)}")
    originals, references = df["original"].to_list(), df["reference"].to_list()
    metadata = [{"lang": lang} for _, lang in df["lang"].items()]
    return originals, references, metadata


def import_ingredients_complete(
    path: Path,
    start_from: int = 0
) -> Tuple[Iterable[str], Iterable[str], Iterable[Mapping]]:
    """Load Ingredients complete dataset to evaluate Spellcheck on False Positives.

    Args:
        path (Path): Parquet file
        start_from (int, optional): Row index to load the dataset from.. Defaults to 0.

    Returns:
        Tuple[Iterable[str], Iterable[str], Iterable[Mapping]]: Orginals, References, and Metadata
    In this case, References are considered identical to Originals
    """
    if path.suffix != ".parquet":
        raise ValueError(f"Wrong file format. Parquet required. Instead {path.suffix} provided")
    df = pd.read_parquet(path)
    df = df.iloc[start_from:]
    LOGGER.info(f"Data features: {df.columns}")
    LOGGER.info(f"Data length: {len(df)}")
    originals = df["ingredients_text"].to_list()
    references = originals.copy() # Reference =  Original, which means the original is considered as perfect (no error to correct)
    metadata = [{"lang": lang, "code": code} for lang, code in zip(df["lang"], df["code"])]
    return originals, references, metadata


if __name__ == "__main__":
    main()
