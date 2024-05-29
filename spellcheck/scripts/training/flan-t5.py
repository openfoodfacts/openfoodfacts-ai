"""Flan-T5 training script."""
import os
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

# from spellcheck.evaluation.evaluator import SpellcheckEvaluator 


disable_caching()

logger = logging.basicConfig(
    level=logging.getLevelName(logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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
    parser.add_argument("--fp16", type=strtobool, default=False, help="Whether to use bf16.")
    parser.add_argument("--generation_max_tokens", type=int, default=512, help="Max tokens used for text generation in the Trainer module.")
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

    padding = "max_length"                               # Padding configuration. "max_length" means the moodel maxm length
    instruction = "Correct the list of ingredients: "    # Flan-T5 was pretrained using an instruction. We reuse the same

    def train(self, args):

        # Load tokenizer and model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)

        # Load data
        train_dataset = load_dataset(path=args.training_data, split="train") 
        benchmark_dataset = load_dataset(path=args.evaluation_data, split="train")

        #TODO: Solve dependency issue
        # # Evaluation
        # evaluator = SpellcheckEvaluator(benchmark_dataset["original"], beta=args.beta)

        # def compute_metrics(eval_preds: Tuple) -> Mapping[str, Any]:
        #     """Metrics calculation used by Trainer.
        #     The function needs to be nested into the training function because of CometML, 
        #     which loses tracking otherwise.

        #     Args:
        #         eval_preds (Tuple): Containing prediction and label tokens 

        #     Returns:
        #         Mapping: Metrics on the evaluation dataset
        #     """
        #     # Get the experiment initialized by Trainer
        #     experiment = comet_ml.get_global_experiment()

        #     pred_tokens, ref_tokens = eval_preds

        #     # Need to convert -100 tokens back to pad_token
        #     ref_tokens = np.where(ref_tokens != -100, ref_tokens, tokenizer.pad_token_id)
        #     pred_tokens = np.where(pred_tokens != -100, pred_tokens, tokenizer.pad_token_id)
            
        #     # Transform back in texts
        #     predictions = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        #     references = tokenizer.batch_decode(ref_tokens, skip_special_tokens=True)
            
        #     # Usage of our custom evaluation algorithm
        #     metrics = evaluator.evaluate(predictions=predictions, references=references)
        #     logging.info(f"Metrics: {metrics}")

        #     if experiment:
        #         experiment.log_metrics(metrics)
        #     else:
        #         logging.warning("No experiment in compute_metrics.")
        #     return metrics

        # Prepare data for training
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
        preprocessed_benchmark_dataset = benchmark_dataset.map(
            self.preprocess,
            batched=True,
            remove_columns=benchmark_dataset.column_names,
            fn_kwargs={
                "input_name": "original", 
                "target_name": "reference",
                "tokenizer": tokenizer,
            }
        )

        # Ignore tokenizer pad_token in the loss compuation
        label_pad_token_id = -100
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8
        )

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
            #Logging & evaluation strategies
            logging_dir                         = f"{args.output_dir}/logs",
            logging_strategy                    = "steps",
            logging_steps                       = 500,
            evaluation_strategy                 = "epoch",
            save_strategy                       = "epoch",
            save_total_limit                    = 2,                                   # Number checkpoints saved at the same
            load_best_model_at_end              = True,
            # metric_for_best_model               = "f1_beta",                           # Metric used to select the best model.
        )
        trainer = Seq2SeqTrainer(
            model           = model,
            args            = training_args,
            data_collator   = data_collator,
            train_dataset   = preprocessed_train_dataset,
            eval_dataset    = preprocessed_benchmark_dataset,
            # compute_metrics = compute_metrics,
        )
        trainer.train()


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