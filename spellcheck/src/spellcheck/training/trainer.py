"""Notes: Use HFArgumentParser in script"""
import os
from abc import ABC, abstractmethod
from typing import (
    Optional, 
    Mapping, 
    List, 
    Union, 
    Any, 
    Iterable, 
    Tuple,
    Type, 
    Dict
)
from functools import partial
from tqdm import tqdm

import torch
from pydantic import BaseModel, Field, ConfigDict
from pydantic.dataclasses import dataclass
from datasets import Dataset
from transformers import (
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import AutoPeftModelForCausalLM
import comet_ml

from spellcheck.utils import get_logger
from spellcheck.evaluation.evaluator import SpellcheckEvaluator
from spellcheck.training.configs import (
    DataProcessingConfig,
    SavingConfig,
    ModelConfig,
    InferenceConfig,
    EvaluationDataFeatures,
    EvaluationConfig,
)


LOGGER = get_logger()


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Datasets:
    training_dataset: Dataset
    evaluation_dataset: Optional[Dataset]


class DataProcessor(ABC, BaseModel):
    """Processing class to transform datasets for the model training and evaluation.
    
    The class is designed to inherite the data processing job adapated to the training algorithm, such as SFT, DPO, Instruction-tuning, and so on...
    """

    batched: bool = Field(default=False)
    
    @abstractmethod
    def _process_fn(
        self, 
        element: Mapping[str, Union[Any, List]],
        text_feature: str,
        label_feature: str,
    ) -> Mapping[str, Union[Any, List]]:
        """Processing function used within the Dataset.map() method from the 'datasets' library.
        
        The control the behavior of this function, one should create a new class that inherates from DataProcessor and build
        its own _processed_fn(). The latest is then used in the process() method from the base model.
        """
        raise NotImplementedError
    
    def process(
        self, 
        datasets: Datasets, 
        data_processing_features: DataProcessingConfig,
    ) -> Datasets:
        """Using the process function associated with the class, performs processing of the 
        provided training and evaluation datasets.

        The evaluation dataset is optional.

        Args:
            datasets (Datasets): training and evaluation datasets (the latest is optional)
            training_processing_features (ProcessingFeatures): _description_
            evaluation_processing_features (Optional[ProcessingFeatures]): Default to None.

        Returns:
            Datasets: _description_
        """
        # Training dataset
        processed_training_dataset = self._map_dataset(
            dataset=datasets.training_dataset,
            text_feature=data_processing_features.train_text_feature,
            label_feature=data_processing_features.train_label_feature,
        )
        # Evaluation dataset
        if not datasets.evaluation_dataset and data_processing_features.eval_text_feature or data_processing_features.eval_label_feature:
            raise ValueError(f"Evaluation processing features provided but no evaluation dataset was provided. Datasets dataclass currently provided: {datasets}")
        elif datasets.evaluation_dataset:
            processed_evaluation_dataset = self._map_dataset(
                dataset=datasets.evaluation_dataset,
                text_feature=data_processing_features.eval_text_feature,
                label_feature=data_processing_features.eval_label_feature,
            )
            return Datasets(
                training_dataset=processed_training_dataset,
                evaluation_dataset=processed_evaluation_dataset
            )
        # If only train dataset
        return Datasets(
            training_dataset=processed_training_dataset
        )
    
    def _map_dataset(
        self,
        dataset: Dataset, 
        text_feature: str, 
        label_feature: str
    ) -> Dataset:
        """"""
        return dataset.map(
            partial(
                self._process_fn,
                text_feature=text_feature,
                label_feature=label_feature,
            ),
            batched=self.batched,
            remove_columns=dataset.column_names,
        )
    
    @abstractmethod
    def process_texts(self, texts: Iterable[str]) -> Iterable[str]:
        raise NotImplementedError
            

class SFTDataProcessor(DataProcessor):
    """Data processing engine for Supervised Fine Tuning training.
    """

    instruction_template: str

    def _process_fn(
        self,
        element: Mapping[str, Union[Any, List]], 
        text_feature: str, 
        label_feature: str
    ) -> Mapping[str, Union[Any, List]]:
        """Prepare data for Instruction fine-tuning using the SFT Trainer.
        
        The latest expects the feature column 'text'.
        The text input and label are concatenated into one instruction-prompt using the 'instruction_template'.

        Args:
            element (Mapping[str, Union[Any, List]]): dictionnary element during the dataset mapping.
            text_feature (str): Text column name in the dataset.
            label_feature (str): Label column name in the dataset

        Returns:
            Mapping[str, Union[Any, List]]: Processed dictionnary.
        """
        instruction = self._prepare_instruction(element[text_feature])
        return {"text": instruction + element[label_feature]}

    def _prepare_instruction(self, text: str) -> str:
        """Prepare instruction based on the instruction-template. 
        This function is primordial for the training step, but also during inference.
        """
        return self.instruction_template.format(text)

    def process_texts(self, texts: Iterable[str]) -> Iterable[str]:
        """"""
        return [self._prepare_instruction(text) for text in texts]
    

class SavingProcessor(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def save(self, trainer: Trainer):
        raise NotImplementedError
    

class LoRASavingProcessor(SavingProcessor):

    output_dir: str
    saving_config: SavingConfig
    
    def save(self, trainer: Trainer):
        """Check these links to know more about saving LoRA adapters after training: 
            * https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
            * https://github.com/philschmid/llm-sagemaker-sample/blob/main/scripts/run_qlora.py
        """
        LOGGER.info(f"Saving tokenizer in {self.output_dir}")
        trainer.tokenizer.save_pretrained(self.output_dir)
        LOGGER.info(f"Saving model and adapters in {self.output_dir}")

        if self.saving_config.merge_weights:
            # Save adapters only
            trainer.model.save_pretrained(self.output_dir)
            # clear memory
            # del model #NOTE: Let's see what does it do since I'm not in the script.
            del trainer
            torch.cuda.empty_cache()

            # Load PEFT model in fp16. It uses the saved LoRA adapters and load the pretrained model for the HF hub.
            LOGGER.info("Load PEFT model.")
            torch_dtype = torch.float16 if self.saving_config.f16 else torch.float32
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                trust_remote_code=True,  # Required because PEFT load pretrained model before merging adapters
            )
            model = model.merge_and_unload()
            # Save merged model
            model.save_pretrained(
                self.output_dir, 
                safe_serialization=True, 
                max_shard_size=self.saving_config.max_shard_size
            )
            LOGGER.info("Model merged and saved succesfully.")
            del model

            # Remove adapters from the directory
            try:
                os.remove(os.path.join(self.output_dir, "adapter_config.json"))
                os.remove(os.path.join(self.output_dir, "adapter_model.safetensors"))
            except Exception as e:
                LOGGER.warning(f"Something went wrong with trying to remove adapters files for the training directory. Error: {e}")

        else:
            # Save adapters only
            trainer.model.save_pretrained(self.output_dir) 


class InferenceProcessor(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    inference_config: InferenceConfig
    data_processor: Optional[DataProcessor] = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def inference(self):
        raise NotImplementedError

    @classmethod
    def load_pretrained(
        cls, 
        model_dir: str,
        model_conf: ModelConfig,
        data_processor: DataProcessor,
        inference_config: InferenceConfig
    ) -> None:
        """"""
        # Prepare the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        torch_dtype = torch.bfloat16 if model_conf.bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=model_conf.device_map,
            attn_implementation=model_conf.attn_implementation,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        return cls(
            tokenizer=tokenizer,
            model=model,
            data_processor=data_processor,
            inference_config=inference_config
        )
    
    def _batch_process(self, lst: Iterable, batch_size: int = 1):
        """"""
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]


class TextGenerationInference(InferenceProcessor):

    inference_config: InferenceConfig

    def inference(
        self, 
        texts: Iterable[str], 
    ) -> Iterable[str]:
        """"""
        predictions = []
        processed_texts = self.data_processor.process_texts(texts)

        if self.inference_config.batch_size > 1:
            processed_texts = self._batch_process(processed_texts, batch_size=self.inference_config.batch_size)
        
        for text_batch in tqdm(
            processed_texts, 
            total=len(texts), 
            desc="Prediction" if self.inference_config.batch_size == 1 else f"Prediction in batch: batch_size = {self.inference_config.batch_size == 1}"
        ):
            encodings = self.tokenizer(
                text_batch, 
                add_special_tokens=True, 
                return_tensors="pt",
                padding="longest", # In batch, required padding strategy between 
            )
            encodings = {k: v.to(self.device) for k,v in encodings}
            pred_encodings = self.model.generate(
                **encodings,
                do_sample=False,
                max_new_tokens=self.inference_config.max_new_tokens,
            )
            # Decode, remove instruction and strip text
            prediction_batch = self._post_process(
                encodings=pred_encodings,
                text_batch=text_batch,
            )
            predictions.extend(prediction_batch)
        return predictions

    def _post_process(
            self, 
            encodings: Mapping, 
            text_batch: Iterable[str]
        ) -> List[str]:
        """"""
        predictions = self.tokenizer.batch_decode(encodings, skip_special_tokens=True)        
        return [prediction[len(text):].strip() for prediction, text in zip(predictions, text_batch)]


class EvaluationProcessor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    evaluator_type: Type[SpellcheckEvaluator]
    inference_processor: InferenceProcessor
    evaluation_dataset: Dataset
    evaluation_features: EvaluationDataFeatures
    evaluation_config: EvaluationConfig

    def model_post_init(self):
        """Prepare evaluator. Pydantic method."""
        orginals, _ = self._prepare_data()
        self.evaluator = self.evaluator_type(originals=orginals)

    def evaluate(self) -> Dict[str, float]:
        """
        """
        # Load texts
        originals, references = self._prepare_data()
        # Predictions
        predictions = self.inference_processor.inference(texts=originals)
        # Evaluation
        metrics = self.evaluator.evaluate(predictions=predictions, references=references)
        LOGGER.info(f"Evaluation metrics: {metrics}")
        # Save predictions
        prediction_dataset = self.evaluation_dataset.add_column(
            name="prediction", 
            column=predictions
        )
        if self.evaluation_config.save_predictions_path:
            LOGGER.info(f"Predictions are saved in: {self.evaluation_config.save_predictions_path}")
            prediction_dataset.save_to_disk(self.evaluation_config.save_predictions_path)
        return metrics
    
    def _prepare_data(self) -> Tuple[Iterable[str], Iterable[str]]:
        """"""
        originals = self.evaluation_dataset[self.evaluation_features.eval_text_feature]
        references = self.evaluation_dataset[self.evaluation_features.eval_label_feature]
        return originals, references


class ExperimentLogger(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def log(self):
        raise NotImplementedError
    

class CometExperimentLogger(ExperimentLogger):
    """"""
    
    experiment: comet_ml.Experiment
    workspace: str = os.getenv("COMET_WORKSPACE_NAME")
    project_name: str = os.getenv("COMET_PROJECT_NAME")
    api_key: str = os.getenv("COMET_API_KEY")
    
    @classmethod
    def load_experiment(cls):
        global_experiment = comet_ml.get_global_experiment()
        experiment = global_experiment if global_experiment else comet_ml.Experiment(
            api_key=cls.api_key,
            project_name=cls.project_name,
            workspace=cls.workspace,
        )
        return cls(experiment=experiment)

    def log(
        self, 
        metrics: Optional[Mapping] = None, 
        parameters: Optional[Mapping] = None, 
        model_uri: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        if metrics:
            self._log_metrics(metrics)
        if parameters:
            self._log_parameters(parameters)
        if model_uri:
            self._log_model(model_uri)
        if tags:
            self._log_tags(tags)

    def _log_metrics(self, **kwargs) -> None:
        self.experiment.log_metrics(kwargs)

    def _log_parameters(self, **kwargs) -> None:
        self.experiment.log_parameters(kwargs)

    def _log_model(self, model_uri: str, artifact_name: str = "model") -> None:
        self.experiment.log_remote_model(
            artifact_name, model_uri, sync_mode=False
        )

    def _log_tags(self, tags: List[str]) -> None:
        self.experiment.add_tags(tags)
