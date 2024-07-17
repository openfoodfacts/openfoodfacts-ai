import os
import argparse
import json
from distutils.util import strtobool
from typing import Mapping, List
from dotenv import load_dotenv

import torch
from datasets import load_dataset, disable_caching
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)
from peft import LoraConfig
from trl import SFTTrainer
import comet_ml

from spellcheck.utils import get_logger
from spellcheck.evaluation.evaluator import SpellcheckEvaluator 


# For testing
load_dotenv()

# Dataset not saved in cache to save time
disable_caching()

LOGGER = get_logger("INFO")

# Retrieve Sagemaker job name to get the model artifact from S3
SM_TRAINING_ENV = json.loads(os.getenv("SM_TRAINING_ENV"))  # Need to be deserialized
SM_JOB_NAME = SM_TRAINING_ENV["job_name"]

# Where the model artifact is stored 
S3_MODEL_URI = os.getenv("S3_MODEL_URI")

# Tags. JSON Serialized as a string because List is not serializable 
EXPERIMENT_TAGS = os.getenv("EXPERIMENT_TAGS").split(",")


def parse_args():
    parser = argparse.ArgumentParser()

    # Sagemaker environment
    parser.add_argument("--training_data", type=str, default=os.getenv("SM_CHANNEL_TRAINING_DATA"))       # "SM_CHANNEL_{name_data}"
    parser.add_argument("--evaluation_data", type=str, default=os.getenv("SM_CHANNEL_EVALUATION_DATA"))
    parser.add_argument("--output_dir", type=str, default=os.getenv("SM_MODEL_DIR"))

    #Training
    parser.add_argument("--pretrained_model_name", type=str, help="Pretrained model id to fine-tune from the Hugging Face Hub.")
    parser.add_argument("--num_train_epochs", type=float, default=1, help="Number of epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of steps used for a linear warmup from 0 to `learning_rate`")
    parser.add_argument("--warmup_ratio", type=float, default=0, help="Warm-up ratio.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay to prevent overfitting")
    parser.add_argument("--gradient_checkpointing", type=strtobool, default=False, help="To reduce GPU memory footprint during training")
    parser.add_argument("--fp16", type=strtobool, default=False, help="Whether to use fp16.")
    parser.add_argument("--bf16", type=strtobool, default=False, help="Whether to use bf16.")
    parser.add_argument("--tf32", type=strtobool, default=False, help="Whether to use tf32.")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning scheduler type.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate bacthes before back propagation.")
    parser.add_argument("--quantize", type=strtobool, default=True, help="Model quantization to save memeory footprint.")
    parser.add_argument("--logging_steps", type=int, default=25, help="Number of steps between training log.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of steps between evaluation computation.")
    parser.add_argument("--save_total_limit", type=int, default=0, help="Number of checkpoint saved at the same time during the training.")
    parser.add_argument("--train_data_revision", type=str, default="v0", help="Revision of the training data on HuggingFace.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length used in TRL.")
    parser.add_argument("--packing", type=strtobool, default=True, help="Pack exemples together during training.")

    # Versions
    parser.add_argument("--training_data_version", type=str, default="v0", help="Training dataset version.")
    parser.add_argument("--evaluation_data_version", type=str, default="v0", help="Evaluation dataset version.")

    # Evaluation
    parser.add_argument("--beta", type=float, default=1, help="Coefficient used in f1-beta score. beta < 1 favors Precision over Recall.")

    args = parser.parse_known_args()
    return args


class LLMQLoRATraining:

    def train(self, args):
        LOGGER.info("Start training.")
        LOGGER.info(f"Training dir: {args.output_dir}")

        LOGGER.info("Load datasets.")
        train_dataset = load_dataset(
            path=args.training_data, 
            split="train+test",
            revision=args.train_data_revision,
        ) 
        evaluation_dataset = load_dataset(path=args.evaluation_data, split="train")
        LOGGER.info(f"Training dataset: {train_dataset}")
        LOGGER.info(f"Evaluation dataset: {evaluation_dataset}")
        
        LOGGER.info("Load tokenizer.")
        # Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        torch_dtype = torch.bfloat16 if args.bf16 else torch.float32 # bf16 required by Flash Attention

        if args.quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        else:
            quantization_config = None

        LOGGER.info("Load model.")
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        
        ################
        # Preprocessing
        ################
        LOGGER.info("Start data preprocessing.")
        preprocessed_train_dataset = train_dataset.map(
            self.preprocess, 
            batched=False, 
            remove_columns=train_dataset.column_names, 
            fn_kwargs={
                "input_name": "original", 
                "target_name": "reference",
            }
        )
        preprocessed_evaluation_dataset = evaluation_dataset.map(
            self.preprocess, 
            batched=False, 
            remove_columns=evaluation_dataset.column_names, 
            fn_kwargs={
                "input_name": "original", 
                "target_name": "reference",
            }
        )
        LOGGER.info(f"Preprocessed_train_data features: {preprocessed_train_dataset}")
        LOGGER.info(f"Fist row of the preprocessed dataset: {preprocessed_evaluation_dataset[0]}")

        ################
        # PEFT
        ################
        # LoRA config based on QLoRA paper & Sebastian Raschka experiment
        peft_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM", #TODO: try SEQ_2_SEQ_LM
        )

        ################
        # Training
        ################
        training_args = TrainingArguments(
            output_dir                          = args.output_dir,                     # Model checkpoints directory
            per_device_train_batch_size         = args.per_device_train_batch_size,
            per_device_eval_batch_size          = args.per_device_eval_batch_size,
            fp16                                = args.fp16,                           # Overflows with fp16
            bf16                                = args.bf16,
            tf32                                = args.tf32,
            learning_rate                       = args.lr,                             # https://huggingface.co/docs/transformers/en/model_doc/t5#training:~:text=Additional%20training%20tips%3A
            num_train_epochs                    = args.num_train_epochs,
            warmup_steps                        = args.warmup_steps,
            warmup_ratio                        = args.warmup_ratio,
            weight_decay                        = args.weight_decay,
            gradient_checkpointing              = args.gradient_checkpointing,
            optim                               = args.optim,                          # AdamW or AdaFactor        
            lr_scheduler_type                   = args.lr_scheduler_type,
            gradient_accumulation_steps         = args.gradient_accumulation_steps,         
            #Logging & evaluation strategies
            logging_dir                         = f"{args.output_dir}/logs",
            logging_strategy                    = "steps",
            logging_steps                       = args.logging_steps,
            evaluation_strategy                 = "steps",
            eval_steps                          = args.eval_steps,
            save_strategy                       = "steps",
            save_total_limit                    = args.save_total_limit,
            load_best_model_at_end              = True,
            # metric_for_best_model               = "f1_beta",                           # Metric used to select the best model.
            report_to="comet_ml",
        )
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=preprocessed_train_dataset,
            eval_dataset=preprocessed_evaluation_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length, # Prompt instruction + Text
            packing=args.packing,
            dataset_text_field="text",
            dataset_kwargs={
                "add_special_tokens": True,  # Add bos token and other special token from the tokenizer
                "append_concat_token": True,  # If true, appends eos_token_id at the end of each sample being packed.
            },
        )
        LOGGER.info("Start training.")
        trainer.train()

        #############
        # SAVE MODEL
        #############
        LOGGER.info("Start saving.")
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(training_args.output_dir)
        trainer.tokenizer.save_pretrained(training_args.output_dir)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # list file in output_dir
        print(os.listdir(training_args.output_dir))

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,  # Required because PEFT load pretrained model before merging adapters
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()
        model.save_pretrained(
            args.output_dir, safe_serialization=True, max_shard_size="2GB"
        )
        # clear memory because of next evaluation step
        del model

        #############
        # EVALUATE
        #############
        LOGGER.info("Start evaluation.")
        # This process is required since CometML shuts down connection to the experiment run after training
        experiment = comet_ml.get_global_experiment()
        LOGGER.info(f"Experiment name after Transformers trainer: {experiment.get_name()}")
        experiment = comet_ml.ExistingExperiment(experiment_key=experiment.get_key())

        # Load fine-tuned model 
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        LOGGER.info("Start evaluating model on benchmark.")
        originals = evaluation_dataset["original"]
        references = evaluation_dataset["reference"]
        evaluator = SpellcheckEvaluator(originals=originals, beta=args.beta)
        predictions = self.inference(texts=originals, model=model, tokenizer=tokenizer)
        metrics = evaluator.evaluate(predictions=predictions, references=references)
        LOGGER.info(f"Evaluation metrics: {metrics}")
        experiment.log_metrics(metrics)
        # Save evaluation in S3
        prediction_dataset = evaluation_dataset.add_column(name="prediction", column=predictions)
        s3_evaluation_path = os.path.join(os.getenv("S3_EVALUATION_URI"), "evaluation-" + SM_JOB_NAME)
        LOGGER.info(f"S3 URI where predictions on evaluation are sent to: {s3_evaluation_path}")
        experiment.log_parameter("evaluation_uri", s3_evaluation_path)
        prediction_dataset.save_to_disk(s3_evaluation_path)

        #############
        # ADDITIONAL EXP LOGGING
        #############

        LOGGER.info("Start logging additional info into the experiment tracker.")
        
        # Experiment tags
        LOGGER.info(f"Log tags: {EXPERIMENT_TAGS}")
        experiment.add_tags(EXPERIMENT_TAGS)

        # Log remote model artifact from s3
        model_uri = os.path.join(S3_MODEL_URI, SM_JOB_NAME, "output/model/")
        LOGGER.info(f"Training job uri: {model_uri}")
        experiment.log_remote_model(
            "model", 
            model_uri, 
            sync_mode=False
        )

        # Log dataset lengths
        experiment.log_parameters({
            "training_dataset_length": len(train_dataset),
            "evaluation_dataset_length": len(evaluation_dataset),
        })

        # Log Metaflow run id
        if metaflow_run_id := os.getenv("METAFLOW_RUN_ID"):
            experiment.log_parameter("metaflow_run_id", metaflow_run_id)
        
        # Log training job name
        experiment.log_parameter("training_job_name", SM_JOB_NAME)

        LOGGER.info("Training job finished.")


    def preprocess(
            self, sample: Mapping, input_name: str, target_name: str, 
        ) -> Mapping:
        """Preprocess dataset using the `map()` function.

        Args:
            sample (Mapping): Batch of the dataset
            input_name (str): Model training text input feature.*
            target_name (str): Model training text label feature. 

        Returns:
            Mapping: User/Assistant exchanges used for Instruction Fine-Tuning. Used with tokenizer.apply_chat_template()
        """
        instruction = self.prepare_instruction(sample[input_name])
        return {"text": instruction + sample[target_name]}

    def prepare_instruction(self, text: str) -> str:
        """Prepare instruction prompt for fine-tuning and inference.

        Args:
            text (str): List of ingredients

        Returns:
            str: Instruction.
        """
        instruction = (
            "###Correct the list of ingredients:\n"
            + text
            + "\n\n###Correction:\n"
        )
        return instruction

    def inference(
        self, 
        texts: List[str], 
        model: AutoModelForCausalLM, 
        tokenizer: PreTrainedTokenizerBase,
        **gen_kwargs,
    ) -> List[str]:
        """Use fine-tuned model for inference.

        NOTE: Inference extremely long because unoptimized (~20-30min). Thinking about using vLLM in another environment for evaluation.  

        Args:
            texts (List[str]): Sequence of lists of ingredients.
            model (AutoModelForCausalLM): LLM
            tokenizer (PreTrainedTokenizerBase): Tokenizer for causal LM

        Returns:
            List[str]: Sequence of predictions.
        """
        predictions = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            for text in tqdm(texts, total=len(texts), desc="Prediction"):
                prompt = self.prepare_instruction(text)
                input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
                pred = model.generate(
                    input_ids.to(device),
                    # repetition_penalty=1.03,
                    do_sample=False,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id,
                    **gen_kwargs,
                )
                # Decode, remove instruction and strip text
                prediction = tokenizer.decode(pred[0], skip_special_tokens=True)[len(prompt):].strip()
                predictions.append(prediction)
        return predictions


if __name__ == "__main__":
    args, _ = parse_args()
    LLMQLoRATraining().train(args)
