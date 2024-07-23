from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataProcessingConfig:
    batched: bool = field(default=False)


class SFTDataProcessingConfig(DataProcessingConfig):
    instruction_template: str = "###Correct the list of ingredients:\n{}\n\n###Correcton:\n" #TODO: add jinja instruction

@dataclass
class ModelConfig:
    pretrained_model_name: str
    model_bf16: bool = field(default=False)
    device_map: str = field(default="auto")
    attn_implementation: str = field(default="flash_attention_2")


@dataclass
class DataConfig:
    training_data: str = field(default="openfoofacts/spellcheck-dataset")
    evaluation_data: str = field(default="openfoodfacts/spelcheck-benchmark")
    train_split: str = field(default="train")
    eval_split: str = field(default="train")
    train_data_version: str = field(default="v0.0")
    eval_data_version: str = field(default="v0.0")
    train_data_revision: Optional[str] = field(default=None)
    eval_data_revision: str = field(default=None)


@dataclass
class TrainingConfig:
    output_dir: str = field(default="model_dir")


@dataclass
class SavingConfig:
    merge_weights: bool = field(default=False)
    max_shard_size: str = field(default="2GB")
    f16: bool = field(default=True)


@dataclass
class InferenceConfig:
    max_new_tokens: int = field(default=1024)
    batch_size: int = field(default=1)


@dataclass
class TrainingDataFeatures:
    train_text_feature: str = field(default="text")
    train_label_feature: str = field(default="label")


@dataclass
class EvaluationDataFeatures:
    eval_text_feature: str = field(default="text")
    eval_label_feature: str = field(default="label")

@dataclass
class EvaluationConfig:
    save_predictions_path: Optional[str] = field(default=None)


