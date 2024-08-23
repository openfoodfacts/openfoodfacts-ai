"""Using the existing benchmark and evaluation algorithm, evaluate exisiting LLM APIs
* OpenAI GPT-3.5, GPT-4
* Gemini
* Claude 3
* ...
"""

from dotenv import load_dotenv

from spellcheck.utils import get_repo_dir, get_logger
from spellcheck.spellcheck import Spellcheck
from spellcheck.model import (
    AnthropicChatCompletion, 
    OpenAIChatCompletion, 
    RulesBasedModel, 
    GeminiModel, 
    LLMInferenceEndpoint,
)
from spellcheck.prompt import SystemPrompt, Prompt
from spellcheck.argilla.deployment import BenchmarkEvaluationArgilla, IngredientsCompleteEvaluationArgilla
from spellcheck.evaluation.evaluation import Evaluate, import_benchmark, import_ingredients_complete


REPO_DIR = get_repo_dir()
BENCHMARK_PATH = REPO_DIR / "data/benchmark/verified_benchmark.parquet"
INGREDIENTS_COMPLETE_DATA_PATH = REPO_DIR / "data/database/ingredients_complete.parquet"

# Metrics
METRICS_PATH = REPO_DIR / "data/evaluation/metrics.jsonl"

MODEL_NAME = "gpt-4o-2024-05-13"
BENCHMARK_VERSION = "v7.3"
PROMPT_VERSION = "v7"
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
ARGILLA_INGREDIENTS_COMPLETE_DATASET_NAME = f"Evaluation-{MODEL_NAME}-ingredients-complete-{INGREDIENTS_COMPLETE_VERSION}-prompt-{PROMPT_VERSION}".replace(".", "")

LOGGER = get_logger("INFO")

load_dotenv()


def main():
    spellcheck=Spellcheck(
        model=OpenAIChatCompletion(
            prompt_template=Prompt.spellcheck_prompt_template, #If Claude, use custom prompt template
            system_prompt=SystemPrompt.spellcheck_system_prompt,
            model_name=MODEL_NAME,
        )
    )

    ####################### Evaluate on benchmark
    originals, references, metadata = import_benchmark(
        path=BENCHMARK_PATH,
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
    evaluation.compute_metrics()
    # Human evaluation
    BenchmarkEvaluationArgilla.from_jsonl(
        path=PREDICTIONS_EVALUATION_PATH
    ).deploy(
        dataset_name=ARGILLA_BENCHMARK_DATASET_NAME)
    

    # ####################### Evaluate on Ingredient complete dataset
    # originals, references, metadata = import_ingredients_complete(path=INGREDIENTS_COMPLETE_DATA_PATH)
    # evaluation = Evaluate(
    #     model_name=MODEL_NAME,
    #     metrics_path=METRICS_PATH,
    #     benchmark_version=INGREDIENTS_COMPLETE_VERSION,
    #     prompt_version=PROMPT_VERSION,
    #     predictions_path=PREDICTION_INGREDIENTS_COMPLETE_PATH
    # )
    # evaluation.run_evaluation(
    #     originals=originals,
    #     references=references,
    #     spellcheck=spellcheck,
    #     metadata=metadata,
    #     wait=WAIT
    # )
    # IngredientsCompleteEvaluationArgilla.from_jsonl(
    #     path=PREDICTION_INGREDIENTS_COMPLETE_PATH
    # ).deploy(
    #     dataset_name=ARGILLA_INGREDIENTS_COMPLETE_DATASET_NAME
    # )


if __name__ == "__main__":
    main()
