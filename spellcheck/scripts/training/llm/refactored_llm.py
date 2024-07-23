import os
import json
from dotenv import load_dotenv

import torch
from datasets import(
    load_dataset,
    disable_caching,
)
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from spellcheck.utils import get_logger
from spellcheck.training.configs import (
    DataProcessingConfig,
    ModelConfig,
    DataConfig,
    SavingConfig,
    InferenceConfig,
    TrainingDataFeatures,
    EvaluationDataFeatures,
    EvaluationConfig,
)
from spellcheck.training.trainer import (
    Datasets,
    SFTDataProcessor,
    LoRASavingProcessor,
    TextGenerationInference,
    CometExperimentLogger,
    EvaluationProcessor,
    TextGenerationInference,
)
from spellcheck.training.utils import CustomCometCallback
from spellcheck.evaluation.evaluator import SpellcheckEvaluator


LOGGER = get_logger("INFO")

# For debugging
load_dotenv()

# Dataset not saved in cache to save time
disable_caching()


def main():
    LOGGER.info("Start training job.")

    ######################
    # SETUP
    ######################
    LOGGER.info("Parse information from CLI using Argparser.")
    parser = HfArgumentParser([
        SFTConfig,
        BitsAndBytesConfig,
        LoraConfig,
        DataProcessingConfig,
        TrainingDataFeatures,
        EvaluationDataFeatures,
        ModelConfig,
        DataConfig,
        SavingConfig,
        InferenceConfig,
        EvaluationConfig,
    ])
    LOGGER.info(f"Parsed dataclasses: {parser.parse_args_into_dataclasses()}")
    (
        sft_config, 
        quantization_config, 
        lora_config, 
        data_processing_config,
        training_data_features,
        evaluation_data_features,
        model_config, 
        data_config, 
        saving_config, 
        inference_config,
        evaluation_config,
    ) = parser.parse_args_into_dataclasses()

    # Sagemaker environment variables: https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    OUTPUT_DIR = os.getenv("SM_MODEL_DIR")
    SM_TRAINING_ENV = json.loads(os.getenv("SM_TRAINING_ENV"))  # Need to be deserialized
    SM_JOB_NAME = SM_TRAINING_ENV["job_name"]
    # Where the model artifact is store. Can be compressed (model.tar.gz) or decompressed (model/)
    S3_MODEL_URI = os.path.join(os.getenv("S3_MODEL_URI"), "output/model/")

    # CometML tags. JSON Serialized as a string because List is not serializable 
    EXPERIMENT_TAGS = os.getenv("EXPERIMENT_TAGS").split(",")

    ######################
    # LOAD DATA
    ######################
    LOGGER.info("Load datasets.")
    datasets = Datasets(
        training_dataset=load_dataset(
            path=data_config.training_data, 
            split=data_config.train_split,
            revision=data_config.train_data_revision,
        ),
        evaluation_dataset=load_dataset(
            path=data_config.evaluation_data,
            split=data_config.eval_split,
            revision=data_config.eval_data_revision,
        )
    )
    LOGGER.info(f"Training dataset: {datasets.training_dataset}")
    LOGGER.info(f"Evaluation dataset: {datasets.evaluation_dataset}")

    ######################
    # DATA PROCESSING
    ######################
    LOGGER.info("Start pre-processing datasets.")
    data_processor = SFTDataProcessor(
        batched=False,
        instruction_template="###Correct the list of ingredients:\n{}\n\n###Correcton:\n" #TODO
    )
    processed_datasets = data_processor.process(
        datasets=datasets,
        data_processing_config=data_processing_config
    )
    LOGGER.info(f"Processed training dataset: {processed_datasets.training_dataset}")
    LOGGER.info(f"Processed evaluation dataset: {processed_datasets.evaluation_dataset if processed_datasets.evaluation_dataset else "No evaluation dataset provided."}")

    ######################
    # MODEL PREPARATION
    ######################
    LOGGER.info("Start preparing the tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    LOGGER.info("Start preparing the model.")
    torch_dtype = torch.bfloat16 if model_config.bf16 else torch.float32 # bf16 required by Flash Attention
    model = AutoModelForCausalLM.from_pretrained(
        model_config.pretrained_model_name,
        device_map=model_config.device_map,
        torch_dtype=torch_dtype,
        attn_implementation=model_config.attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    LOGGER.info("Model prepared succesfull for training.")

    ######################
    # TRAIN
    ######################
    LOGGER.info("Start training.")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=processed_datasets.training_dataset,
        eval_dataset=processed_datasets.evaluation_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )
    # There is an issue when using CometML with Trainer (starts a new experiment)
    # Thus we modified the callback to track experimentation on existing experiment
    trainer.add_callback(CustomCometCallback)
    trainer.train()
    
    ######################
    # SAVING
    ######################
    LOGGER.info("Start saving.")
    saving_processor = LoRASavingProcessor(
        output_dir=OUTPUT_DIR,
        saving_config=saving_config,
    )
    saving_processor.save(trainer=trainer)
    LOGGER.info(f"Model saved at: {saving_processor.output_dir}")

    ######################
    # EVALUATION
    ######################
    LOGGER.info("Start evaluation.")
    inference_processor = TextGenerationInference.load_pretrained(
        model_dir=OUTPUT_DIR,
        model_config=model_config,
        data_processor=data_processor,
        inference_config=inference_config,
    )
    evaluation_processor = EvaluationProcessor(
        evaluator_type=SpellcheckEvaluator,
        inference_processor=inference_processor,
        evaluation_dataset=Datasets.evaluation_dataset,
        evaluation_features=evaluation_data_features,
        evaluation_config=evaluation_config,
    )
    metrics = evaluation_processor.evaluate()
    LOGGER.info(f"Evaluation metrics: {metrics}")

    ######################
    # EXPERIMENTATION LOGGING
    ######################
    LOGGER.info("Start logging additional metrics and parameters to the experiment tracker.")
    experiment_logger = CometExperimentLogger.load_experiment()
    experiment_logger.log(
        tags=EXPERIMENT_TAGS,
        metrics=metrics,
        model_uri=S3_MODEL_URI,
        parameters={
            "training_job_name": SM_JOB_NAME
        }
    )

    LOGGER.info("End of the training job")

if __name__ == "__main__":
    main()
