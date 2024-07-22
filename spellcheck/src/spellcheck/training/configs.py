from typing import Optional
from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class DataProcessingConfig:
    batched: bool = Field(default=False)


@dataclass
class ModelConfig:
    pretrained_model_name: str = Field(description="Pretrained model on Hugging Face.")
    bf16: bool = Field(default=True)
    device_map: str = Field(default="auto")
    attn_implementation: str = Field(default="flash_attention_2")


@dataclass
class DataConfig:
    training_data: str = Field(default="openfoofacts/spellcheck-dataset")
    evaluation_data: str = Field(default="openfoodfacts/spelcheck-benchmark")
    train_split: str = Field(default="train")
    eval_split: str = Field(default="train")
    train_data_version: str = Field(default="v0.0")
    eval_data_version: str = Field(default="v0.0")
    train_data_revision: str = Field(default="v0")
    eval_data_revision: str = Field(default="v0")


@dataclass
class TrainingConfig:
    output_dir: str = Field("")


@dataclass
class SavingConfig:
    merge_weights: bool = Field(default="False")
    max_shard_size: str = Field(default="2GB")
    f16: bool = Field(default=True)


@dataclass
class InferenceConfig:
    max_new_tokens: int = Field(default=1024)
    batch_size: int = Field(default=1)


@dataclass
class TrainingDataFeatures:
    train_text_feature: str = Field(default="text")
    train_label_feature: str = Field(default="label")


@dataclass
class EvaluationDataFeatures:
    eval_text_feature: str = Field(default="text")
    eval_label_feature: str = Field(default="label")

@dataclass
class EvaluationConfig:
    save_predictions_path: Optional[str] = Field(default=None, description='Can be S3 uri')


