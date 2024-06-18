"""Evaluation module."""
from pathlib import Path
from typing import Iterable, Mapping, Tuple
import time
from tqdm import tqdm
import json
from datetime import datetime

import pandas as pd

from spellcheck.spellcheck import Spellcheck
from spellcheck.utils import get_logger
from spellcheck.evaluation.evaluator import SpellcheckEvaluator


LOGGER = get_logger()


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
    if path.suffix != ".parquet":
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

   
