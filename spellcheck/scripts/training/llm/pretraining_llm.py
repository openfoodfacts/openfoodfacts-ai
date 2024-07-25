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
import comet_ml

from spellcheck.utils import get_logger
from spellcheck.training.configs import (
    ModelConfig,
    DataConfig,
    SavingConfig,
)
from spellcheck.training.trainer import (
    Datasets,
    LoRASavingProcessor,
    CometExperimentLogger,
)
from spellcheck.training.utils import CustomCometCallback


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
        # BitsAndBytesConfig,
        # LoraConfig,
        ModelConfig,
        DataConfig,
        SavingConfig,
    ])
    (
        sft_config, 
        # quantization_config, 
        # lora_config, 
        model_config, 
        data_config, 
        saving_config, 
    ) = parser.parse_args_into_dataclasses()

    #NOTE: Bug with LoraConfig and HFArgumentParser (Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union` because the argument parser only supports one type per argument. Problem encountered in field 'init_lora_weights'.)
    # We instantiate LoraConfig "manually"
    lora_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    torch_dtype = torch.bfloat16 if sft_config.bf16 else torch.float32

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )

    # Sagemaker environment variables: https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    OUTPUT_DIR = os.getenv("SM_MODEL_DIR")
    SM_TRAINING_ENV = json.loads(os.getenv("SM_TRAINING_ENV"))  # Need to be deserialized
    SM_JOB_NAME = SM_TRAINING_ENV["job_name"]
    # Where the model artifact is store. Can be compressed (model.tar.gz) or decompressed (model/)
    S3_MODEL_URI = os.path.join(os.getenv("S3_MODEL_URI"), "output/model/")

    #Comet experiment
    EXPERIMENT_KEY = os.getenv("COMET_EXPERIMENT_KEY")
    experiment = comet_ml.ExistingExperiment(previous_experiment=EXPERIMENT_KEY) if EXPERIMENT_KEY else comet_ml.Experiment()

    ######################
    # LOAD DATA
    ######################
    LOGGER.info("Load datasets.")
    training_dataset = load_dataset(
        path=data_config.training_data, 
        split=data_config.train_split,
        revision=data_config.train_data_revision,
    )
    datasets = Datasets(
        training_dataset=training_dataset,
    )
    LOGGER.info(f"Training dataset: {datasets.training_dataset}")

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
    model = AutoModelForCausalLM.from_pretrained(
        model_config.pretrained_model_name,
        device_map=model_config.device_map,
        torch_dtype=torch_dtype,
        attn_implementation=model_config.attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    LOGGER.info("Model prepared succesfully for training.")

    ######################
    # TRAIN
    ######################
    LOGGER.info("Start training.")
    # Since we parse config using Argument 
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=datasets.training_dataset,
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
    saving_processor.save_trainer(trainer=trainer)
    LOGGER.info(f"Model saved at: {S3_MODEL_URI}")

    ######################
    # EXPERIMENTATION LOGGING
    ######################
    LOGGER.info("Start logging additional metrics and parameters to the experiment tracker.")
    experiment_logger = CometExperimentLogger(experiment=experiment)
    experiment_logger.log(
        model_uri=S3_MODEL_URI,
        model_name="pretrained_model",
        parameters={
            "pretraining_job_name": SM_JOB_NAME,
        }
    )

    LOGGER.info("End of the training job.")

if __name__ == "__main__":
    main()
