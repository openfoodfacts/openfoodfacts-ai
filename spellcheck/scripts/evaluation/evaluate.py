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

import pandas as pd

from utils.utils import get_logger, get_repo_dir
from utils.evaluation import SpellcheckEvaluator
from spellcheck import Spellcheck
from utils.model import AnthropicChatCompletion, OpenAIChatCompletion
from utils.prompt import SystemPrompt, Prompt


REPO_DIR = get_repo_dir()
BENCHMARK_PATH = REPO_DIR / "data/benchmark/verified_benchmark.parquet"

# Metrics
METRICS_PATH = REPO_DIR / "data/evaluation/metrics.jsonl"

# LLM
MODEL_NAME = "gpt-4-turbo"

# Predictions JSONL paths to study the results
PREDICTIONS_EVALUATION_PATH = REPO_DIR / "data/evaluation/" / (MODEL_NAME + ".jsonl")

BENCHMARK_VERSION = "0.2"
START = 0 # To restart the run

LOGGER = get_logger()

load_dotenv()


def main():

    originals, references, metadata = load_benchmark(
        benchmark_path=BENCHMARK_PATH,
        start_from=START
    )
    evaluation = Evaluate(
        metrics_path=METRICS_PATH,
        benchmark_version=BENCHMARK_VERSION,
        predictions_path=PREDICTIONS_EVALUATION_PATH,
    )
    evaluation.run_evaluation(
        originals=originals,
        references=references,
        metadata=metadata,
        spellcheck=Spellcheck(
            model=OpenAIChatCompletion(
                prompt_template=Prompt.spellcheck_prompt_template, #If Claude, use custom prompt template
                system_prompt=SystemPrompt.spellcheck_system_prompt,
                model_name=MODEL_NAME
            )
        ),
        wait=2
    )
    evaluation.compute_metrics(
        predictions_path=PREDICTIONS_EVALUATION_PATH,
        model_name=MODEL_NAME
    )

class Evaluate:
    """Evaluation module to compute the performance of the Spellcheck against the benchmark.

    Args:
        metrics_path (Path): Path where to append the Evaluator metrics as a Mapping object.
            This file contains all the metrics from previous runs.
        benchmark_version (str): Version of the benchmark.
        predictions_path (Path): Path where all predictions against the benchmark are stored for further analysis.
    """
    def __init__(
        self,
        metrics_path: Path,
        benchmark_version: str,
        predictions_path: Path,
    ) -> None:
        self.metrics_path = metrics_path
        self.benchmark_version = benchmark_version
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
            for original, reference, md in zip(originals, references, metadata):
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

    def compute_metrics(
        self,
        predictions_path: Path,
        model_name: str
    ) -> None:
        """From the predictions JSONL containing the Spellcheck predictions, compute the metrics using the evaluation module. 

        Args:
            predictions_path (Path): JSONL file where predictions against the benchmark are stored
            model_name (str): Name of the model, model version, or LLM
        """
        with open(predictions_path, "r") as file:
            lines = file.readlines()
            elements = [json.loads(line) for line in lines]
        originals = [element["original"] for element in elements]
        references = [element["reference"] for element in elements]
        predictions = [element["prediction"] for element in elements]
        evaluator = SpellcheckEvaluator(originals=originals) #TODO Remove the module call from the function 
        metrics = evaluator.evaluate(predictions, references)
        metrics_output = {
            "metrics": metrics,
            "model": model_name,
            "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "benchmark_version": self.benchmark_version,
            "benchmark_size": len(predictions)
        }
        with open(self.metrics_path, "a") as file:
            json.dump(metrics_output, file, indent=4)
            file.write("\n")


def load_benchmark(
    benchmark_path: Path, 
    start_from: int = 0
) -> Tuple[Iterable[str], Iterable[str], Iterable[Mapping]]:
    """Utile function to load the benchmark from the parquet file.
    Also, it is possible a previous didn't go through the entire benchmark for many reasons. In this case, 
    there's the possibility to load the benchmark from a specific index instead. 

    Args:
        benchmark_path (Path): Benchmark parquet file
        start_from (int): Row index to load the benchmark from.
    Returns:
        Tuple[Iterable[str], Iterable[str], Iterable[Mapping]]: Originals, References, Metadata from the benchmark
    """
    # Benchmark
    df = pd.read_parquet(benchmark_path)
    # In case the begininning of the benchmark was already processed
    df = df.iloc[start_from:]
    LOGGER.info(f"Data features: {df.columns}")
    LOGGER.info(f"Data length: {len(df)}")
    originals, references = df["original"].to_list(), df["reference"].to_list()
    metadata = [{"lang": lang} for _, lang in df["lang"].items()]
    return originals, references, metadata


if __name__ == "__main__":
    main()
