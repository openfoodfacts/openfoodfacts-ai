"""Flan-T5 training script."""
import os
import sys
import json
from typing import Mapping, Iterable, Any, Tuple
import argparse
from distutils.util import strtobool
import shutil
import logging

import comet_ml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, disable_caching
import numpy as np

from spellcheck.evaluation.evaluator import SpellcheckEvaluator 


disable_caching()

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],  # Get training logging during training job on Sagemaker
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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
    parser.add_argument("--fp16", type=strtobool, default=False, help="Whether to use bf16.")
    parser.add_argument("--generation_max_tokens", type=int, default=512, help="Max tokens used for text generation in the Trainer module.")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning scheduler type.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0, help="Accumulate bacthes before back propagation.")
    parser.add_argument("--instruction", type=str, default="", help="Flan-T5 instruction.")

    # Versions
    parser.add_argument("--training_data_version", type=str, help="Training dataset version.")
    parser.add_argument("--evaluation_data_version", type=str, help="Evaluation dataset version.")

    # Evaluation
    parser.add_argument("--beta", type=float, default=1, help="Coefficient used in f1-beta score. beta < 1 favors Precision over Recall.")

    args = parser.parse_known_args()
    return args


def copy_files(dir: str, *filenames: Iterable[str]) -> None:
    """Copy additional files into the model.tar.gz artifact. 

    Args:
        path (str): SM_MODEL_DIR / code
    """
    os.makedirs(dir, exist_ok=True)
    for filename in filenames:
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), filename), # Source dir
            os.path.join(dir, filename) # Output_dir
        )


class FlanT5Training:
    """Flan-T5 training. 
    """

    padding = "max_length"                               # Padding configuration. "max_length" means the moodel maxm length

    def train(self, args):
        """Training.

        Args:
            args: Arguments from argparse.
        """
        # Load tokenizer and model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)

        # Load data
        train_dataset = load_dataset(path=args.training_data, split="train+test") 
        evaluation_dataset = load_dataset(path=args.evaluation_data, split="train")
        LOGGER.info(f"Training dataset: {train_dataset}")
        LOGGER.info(f"Evaluation dataset: {evaluation_dataset}")
        
        # Add instruction for Flan-T5
        self.instruction = args.instruction

        # Prepare datasets for training
        preprocessed_train_dataset = train_dataset.map(
            self.preprocess, 
            batched=True, 
            remove_columns=train_dataset.column_names, 
            fn_kwargs={
                "input_name": "text", 
                "target_name": "label",
                "tokenizer": tokenizer,
            }
        )
        preprocessed_evaluation_dataset = evaluation_dataset.map(
            self.preprocess,
            batched=True,
            remove_columns=evaluation_dataset.column_names,
            fn_kwargs={
                "input_name": "original", 
                "target_name": "reference",
                "tokenizer": tokenizer,
            }
        )

        # Custom evaluation
        evaluator = SpellcheckEvaluator(
            originals=evaluation_dataset["original"], 
            beta=args.beta
        )

        def compute_metrics(eval_preds: Tuple) -> Mapping[str, Any]:
            """Metrics calculation used by Trainer.
            The function needs to be nested into the training function because of CometML, 
            which loses tracking otherwise.

            Args:
                eval_preds (Tuple): Containing prediction and label tokens 

            Returns:
                Mapping: Metrics on the evaluation dataset
            """
            # Get the experiment initialized by Trainer
            experiment = comet_ml.get_global_experiment()

            pred_tokens, ref_tokens = eval_preds

            # Need to convert -100 tokens back to pad_token
            ref_tokens = np.where(ref_tokens != -100, ref_tokens, tokenizer.pad_token_id)
            
            # Transform back in texts
            predictions = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
            references = tokenizer.batch_decode(ref_tokens, skip_special_tokens=True)
            LOGGER.info(f"First prediction element: {predictions[0]}\nFirst reference element: {references[0]}")

            # Custom evaluation
            metrics = evaluator.evaluate(predictions=predictions, references=references)
            LOGGER.info(f"Metrics: {metrics}")

            if experiment:
                experiment.log_metrics(metrics)
                # Log texts
                for idx in [5, 50, 100]:
                    experiment.log_text(predictions[idx], metadata={"type": "Prediction"})
                    experiment.log_text(references[idx], metadata={"type": "Reference"})
            else:
                logging.warning("No experiment in compute_metrics.")
            return metrics

        # Ignore tokenizer pad_token in the loss compuation
        label_pad_token_id = -100
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8
        )

        # Gradient checkpointing
        if args.gradient_checkpointing:
            model.use_cache = False
        
        # Training
        training_args = Seq2SeqTrainingArguments(
            output_dir                          = args.output_dir,                     # Model checkpoints directory
            per_device_train_batch_size         = args.per_device_train_batch_size,
            per_device_eval_batch_size          = args.per_device_eval_batch_size,
            predict_with_generate               = True,                                # Required for Seq2Seq
            generation_max_length               = args.generation_max_tokens,          # Default to 20 (depends on the task)
            fp16                                = args.fp16,                           # Overflows with fp16
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
            logging_steps                       = 100,
            evaluation_strategy                 = "epoch",
            save_strategy                       = "epoch",
            save_total_limit                    = 2,                                   # Number checkpoints saved at the same
            load_best_model_at_end              = True,
            # metric_for_best_model               = "f1_beta",                           # Metric used to select the best model.
            report_to="comet_ml",
        )
        trainer = Seq2SeqTrainer(
            model           = model,
            args            = training_args,
            data_collator   = data_collator,
            train_dataset   = preprocessed_train_dataset,
            eval_dataset    = preprocessed_evaluation_dataset,
            compute_metrics = compute_metrics,
        )
        trainer.train()
        LOGGER.info("Training finished.")

        # Run and upload benchmark predictions to S3
        LOGGER.info("Start exporting benchmark predictions to S3.")
        predictions, _, _ = trainer.predict(preprocessed_evaluation_dataset)
        preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        prediction_dataset = evaluation_dataset.add_column(name="prediction", column=preds)
        s3_evaluation_path = os.path.join(os.getenv("S3_EVALUATION_URI"), "evaluation-" + SM_JOB_NAME)
        LOGGER.info(f"S3 URI where predictions on evaluation are sent to: {s3_evaluation_path}")
        prediction_dataset.save_to_disk(s3_evaluation_path)

        # Finish Experiment tracking logging
        LOGGER.info("Start logging additional info into the experiment tracker.")

        # This process is required since the a bug with CometML shuts down connection to the experiment run
        experiment = comet_ml.get_global_experiment()
        LOGGER.info(f"Experiment name after Transformers trainer: {experiment.get_name()}")
        experiment = comet_ml.ExistingExperiment(experiment_key=experiment.get_key())
        
        # Experiment tags
        LOGGER.info(f"Log tags: {EXPERIMENT_TAGS}")
        experiment.add_tags(EXPERIMENT_TAGS)

        # Log remote model artifact from s3
        model_uri = os.path.join(S3_MODEL_URI, SM_JOB_NAME, "output/model.tar.gz")
        LOGGER.info(f"Training job uri: {model_uri}")
        experiment.log_remote_model(
            "flan-t5-small-spellcheck", 
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
        
        LOGGER.info("Training job finished.")

    def preprocess(
            self, 
            sample: Mapping, 
            input_name: str, 
            target_name: str, 
            tokenizer: AutoTokenizer
        ) -> Mapping:
        """Preprocess dataset using the `map()` function.

        Args:
            sample (Mapping): Batch of the dataset
            input_name (str): Model training text input feature.
            target_name (str): Model training text label feature. 
            tokenizer (AutoTokenizer): Tokenizer

        Returns:
            Mapping: Processed batch
        """
        # add prefix to the input for t5
        inputs = [self.instruction + item for item in sample[input_name]]
        # tokenize inputs
        model_inputs = tokenizer(
            inputs, 
            padding=self.padding, 
            truncation=True
        )
        # Tokenize targets
        labels = tokenizer(sample[target_name], padding=self.padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


if __name__ == "__main__":
    
    args, _ = parse_args()
    FlanT5Training().train(args)
    copy_files(
        os.path.join(args.output_dir, "code"), 
        "requirements.txt"
    )