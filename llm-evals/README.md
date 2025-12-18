# llm-evals

llm-evals is a framework for evaluating large language models (LLMs) on various tasks. It provides tools to run evaluation on custom datasets, define tasks, and implement evaluators to measure model performance. The framework is designed to be extensible, allowing users to create their own tasks and evaluation metrics.

It relies on `pydantic` for data validation and schema definition, on `pydantic-ai` for LLM model integration and on `pydantic-eval` as an evaluation engine.

## Installation

You can install llm-evals with `uv`. At the root of your project, run:

```bash
uv install
```

## Usage

To run evaluations using llm-evals, you can use the command line interface (CLI).

### Running on a single sample

You can run a specific task on a single image URL:

```bash
python main.py run-task "https://prices.openfoodfacts.org/img/price-tags/000/070/000070276.webp" --model "google-vertex:gemini-2.5-flash-lite" --task "prices:price_tag_extraction"
```

### Evaluate

To evaluate a model on a specific task using a predefined dataset, you can use the following command:

```bash
python main.py evaluate --model "google-vertex:gemini-2.5-flash-lite" --task "prices:price_tag_extraction"
```

Each task has a specific configuration that defines the dataset, the instruction prompt, the output schema we expect from the model.
The dataset comes with custom evaluators that need to be developed. Evaluators are responsible for comparing the model's output with the ground truth and computing relevant metrics.

A list of available tasks can be found in the [tasks directory](./llm_evals/tasks).

The `evaluate` command has several options to customize the evaluation process:

- `--task` (**required**): Specifies the task to be evaluated. The task should be defined in the tasks directory.
- `--model`: Specifies the model to be evaluated. You can use models from different providers (e.g., Google Vertex AI, OpenAI, etc.) by specifying the provider prefix (e.g., `google-vertex:`, `google-gla:`, `openrouter:`,...). Default is set to `google-vertex:gemini-2.5-flash-lite`.
- `--òutput-mode`: The [output mode](https://ai.pydantic.dev/output/) of the pydantic-ai agent. Besides the output modes supported by pydantic-ai, llm-evals also supports `native+prompted` mode, which use the native mode with the JSON schema appended to the prompt. This mode is useful for models run through vLLM providers that don't provide the description of the output schema to the model.
- `--thinking-config`: Configuration for the model's "thinking" process, which can help improve response quality. If not provided, we keep the model default behavior. It can be a string or an int, depending on the provider.
- `--output-path`: Path to save the evaluation results. If not provided, results will only be printed to the console.
- `--filter`: A query string to filter samples from the dataset before evaluation. This allows you to evaluate only a subset of the data based on specific criteria (see below for mode details).
- `--max-concurrency`: Maximum number of concurrent requests to the LLM API, default to None (no limit).
- `--limit`: Limit the number of samples to evaluate. Only the first `n` samples will be evaluated. By default, all samples are evaluated.
- `--only-errors`: Whether to display only error cases (cases where the LLM failed) in the CLI report, default to false.
- `--include-output`: Whether to display the model output in the CLI report, default to false.
- `--include-durations`: Whether to display the durations for each case in the CLI report, default to false.
- `--include-input`: Whether to display the input in the CLI report, default to false.
- `--include-expected-output`: Whether to display the expected output in the CLI report, default to false.
- `--include-reasons`: Whether to display the reasons for each assertion in the CLI report, default to true.


#### Filtering Samples

During evaluation, you can filter samples based on specific criteria. For example, to evaluate only samples with a `country:es` tag, you can use:

```bash
python main.py evaluate --model "google-vertex:gemini-2.5-flash-lite" --task "prices:price_tag_extraction" --filter "tags == 'country:es'"
```
We use [dictquery](https://github.com/cyberlis/dictquery) under the hood for query parsing and filtering, so you can use any valid `dictquery` expression.

The "real" syntax for filtering tags is actually ```--filter "\`metadata.tags.name\` == 'country:es'"```, but we provide `tags ==` as a convenient shortcut. Note that in the full syntax:

- we need to escape the \` character in bash.
- we use `metadata.tags.name` instead of `metadata.tags`. Due to dictquery syntax limitations, we need to embed each tag in a dictionary with a `name` key, which is done at runtime by the library. The same must be done when doing the query.

In short, we advice to use the `tags` shorthand out of convenience.

To filter by case name (usually it's the ID), you can use `--filter "name == '89688'"`.

For example, to filter price tags from France or the Netherlands that are of type 'product', you can use ```tags == 'type:product' AND (tags == 'country:nl' or tags == 'country:fr')```.