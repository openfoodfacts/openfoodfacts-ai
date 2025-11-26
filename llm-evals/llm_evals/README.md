# llm-evals

llm-evals is a framework for evaluating large language models (LLMs) on various tasks. It provides tools to run evaluation on custom datasets, define tasks, and implement evaluators to measure model performance. The framework is designed to be extensible, allowing users to create their own tasks and evaluation metrics.

It relies on `pydantic` for data validation and schema definition, on `pydantic-ai` for LLM model integration and on `pydantic-eval` as an evaluation engine.

## Installation

You can install llm-evals with `uv`. At the root of your project, run:

```bash
uv install
```

## Usage

To run evaluations using llm-evals, you can use the command line interface (CLI). Here is an example command to run evaluations on a specific task:

```bash
python main.py evaluate --model "google-vertex:gemini-2.5-flash-lite" --task "food:product_info_extraction"
```

You can also run a specific task on a single image URL:

```bash
python main.py run-task "https://images.openfoodfacts.org/images/products/932/721/500/0085/1.jpg" --model "google-vertex:gemini-2.5-flash-lite" --task "food:product_info_extraction"
```