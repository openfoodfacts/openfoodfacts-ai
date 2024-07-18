from typing import Dict
from tqdm import tqdm

import metaflow
from datasets import Dataset

from spellcheck.utils import get_logger
from spellcheck.spellcheck import Spellcheck
from spellcheck.model import OpenAIChatCompletion
from spellcheck.prompt import Prompt, SystemPrompt
from spellcheck.argilla.deployment import BenchmarkArgilla


LOGGER = get_logger("INFO")


class GenerateBenchmarkPipeline(metaflow.FlowSpec):

    features = ["original", "reference", "lang", "code"]

    benchmark_path = metaflow.Parameter(
        name="benchmark_path",
        default="data/benchmark/new_benchmark.parquet",
        type=str,
        help="Path of the benchmark"
    )

    model_name = metaflow.Parameter(
        name="model_name",
        default="gpt-3.5-turbo",
        type=str,
        help="LLM name. Need to be the official model endpoint name."
    )

    argilla_dataset_name = metaflow.Parameter(
        name="argilla_dataset_name",
        required=True,
        type=str,
        help="Argilla dataset name during deployment."
    )

    argilla_workspace_name = metaflow.Parameter(
        name="argilla_workspace_name",
        default="spellcheck",
        type=str,
        help="Argilla workspace name."
    )

    @metaflow.step
    def start(self):
        self.benchmark_data = Dataset.from_parquet(self.benchmark_path)
        LOGGER.info(f"Load benchmark as a dataset: {self.benchmark_data}")
        # Check if all features are in the dataset
        for feature in self.features:
            if feature not in self.benchmark_data.column_names:
                raise ValueError(f"The feature {feature} is not in the dataset features: {self.benchmark_data.column_names}")
        self.next(self.generate_synthetic_corrections)

    @metaflow.step
    def generate_synthetic_corrections(self):
        """"""
        spellcheck = Spellcheck(
            model=OpenAIChatCompletion(
                prompt_template=Prompt.spellcheck_prompt_template,
                system_prompt=SystemPrompt.spellcheck_system_prompt,
                model_name=self.model_name
            )
        )

        def process_fn(element: Dict) -> Dict:
            """_summary_

            Args:
                element (Dict): _description_

            Returns:
                Dict: _description_
            """
            if not element["reference"]:
                element["reference"] = spellcheck.correct(element["original"]) #TODO: OpenAI processing (tqdm) not showing
            return element

        self.processed_benchmark = self.benchmark_data.map(
            process_fn, 
            batched=False,
        )
        self.next(self.push_to_argilla)

    @metaflow.step
    def push_to_argilla(self):
        """"""
        BenchmarkArgilla(
            originals=self.processed_benchmark["original"],
            references=self.processed_benchmark["reference"],
            metadata=[
                {"lang": element["lang"], "code": element["code"]} 
                for element in self.processed_benchmark
            ]
        ).deploy(
            dataset_name=self.argilla_dataset_name,
            workspace_name=self.argilla_workspace_name
        )
        self.next(self.end)

    @metaflow.step
    def end(self):
        LOGGER.info("Pipeline finished succesfully.")


if __name__ == "__main__":
    GenerateBenchmarkPipeline()
