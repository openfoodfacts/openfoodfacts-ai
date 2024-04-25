"""Using the existing benchmark and evaluation algorithm, evaluate exisiting LLM APIs
* OpenAI GPT-3.5
* OpenAI GPT-4
* Gemini
"""

from typing import Iterable, Mapping
import json
from pathlib import Path
from dotenv import load_dotenv
import time

import pandas as pd

from utils.utils import get_logger, get_repo_dir
from utils.evaluation import SpellcheckEvaluator, Evaluator
from spellcheck import Spellcheck
from utils.model import OpenAIChatCompletion, AnthropicChatCompletion
from utils.prompt import SystemPrompt, Prompt


REPO_DIR = get_repo_dir()
BENCHMARK_PATH = REPO_DIR / "data/benchmark/verified_benchmark.parquet"
SYNTHETIC_BENCHMARK = REPO_DIR / "data/benchmark/benchmark.json"

# Output paths
GPT35_OUTPUT_EVALUATION_PATH = REPO_DIR / "data/evaluation/gpt_35.json"
GPT4_OUTPUT_EVALUATION_PATH = REPO_DIR / "data/evaluation/gpt_4.json"
CLAUDE3_HAIKU_OUTPUT_EVALUATION_PATH = REPO_DIR / "data/evaluation/claude_3_haiku.json"

BETA_F1 = 1.5  # Favor Precision over Recall

LOGGER = get_logger()

load_dotenv()


def main():

    # Benchmark
    df = pd.read_parquet(BENCHMARK_PATH)
    LOGGER.info(f"Data features: {df.columns}")
    LOGGER.info(f"Data length: {len(df)}")
    originals, references = df["original"].to_list(), df["reference"].to_list()
    metadata = {"lang": df["lang"].to_list()}

    # Initialize
    evaluator = SpellcheckEvaluator(
        originals=originals, beta=BETA_F1
    )
    gpt35_spellcheck = Spellcheck(
        model=OpenAIChatCompletion(
            model_name="gpt-3.5-turbo",
            prompt_template=Prompt.spellcheck_prompt_template,
            system_prompt=SystemPrompt.spellcheck_system_prompt
        )
    )
    claude3_haiku_spellcheck = Spellcheck(
        model=AnthropicChatCompletion(
            prompt_template=Prompt.claude_spellcheck_prompt_template, # Mdofiied prompt for Claude
            system_prompt=SystemPrompt.spellcheck_system_prompt,
            model_name="claude-3-haiku-20240307"
        )
    )
    # GPT-3.5 Chat Completion
    # evaluate(
    #     texts=originals,
    #     references=references,
    #     spellcheck=gpt35_spellcheck,
    #     evaluator=evaluator,
    #     output_path=GPT35_OUTPUT_EVALUATION_PATH,
    #     metadata=metadata
    # )

    # Claude 3 Haiku
    evaluate(
        texts=originals,
        references=references,
        spellcheck=claude3_haiku_spellcheck,
        evaluator=evaluator,
        output_path=CLAUDE3_HAIKU_OUTPUT_EVALUATION_PATH,
        metadata=metadata,
        wait=12 # Free tier limited to 5 requests per minute
    )


def evaluate(
    texts: Iterable[str],
    references: Iterable[str],
    spellcheck: Spellcheck, 
    evaluator: Evaluator, 
    output_path: Path,
    metadata: Mapping = None,
    wait: int = None
):
    """Evaluate Spellcheck modules.

    Args:
        texts (Iterable[str]): Lists of ingredients
        references (Iterable[str]): Benchmark references
        spellcheck (Spellcheck): Spellcheck module
        evaluator (Evaluator): Evaluation algorithm
        output_path (Path): Save results with predictions
        metadata (Mapping, optional): Additional metadata to save. Defaults to None.
        wait (int, optional): Waiting time in case of number of requests per minute limited. Defaults to None.
    """
    predictions = []
    for text in texts:
        predictions.append(spellcheck.correct(text))
        # In case Requests Per Minute are limited 
        if wait:
            time.sleep(wait)

    metrics = evaluator.evaluate(predictions, references)
    output = {
        "metrics": metrics,
        "originals": texts,
        "references": references,
        "predictions": predictions,
        "metadata": metadata
    }
    with open(output_path, "w") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    LOGGER.info(f"Evaluation over and saved at {output_path}.")


if __name__ == "__main__":
    main()
