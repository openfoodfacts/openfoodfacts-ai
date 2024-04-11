"""
This script aims at detecting errors in the training split of the dataset by
performing a cross-validation training, where the training split is divided
into `cross_validation_num` folds. Each fold is used as a test set once, while
the remaining folds are used as the training set. The model is trained on each
fold and evaluated on the corresponding test set. The script logs the
predictions of the model on the test set to Argilla.
"""

import functools
import os

import argilla as rg
import torch
import typer
from datasets import DatasetDict, load_dataset
from tokenizers.pre_tokenizers import Metaspace, Punctuation, Sequence, WhitespaceSplit
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from train import (
    compute_metrics,
    id2label,
    label2id,
    save_prediction_artifacts,
    tokenize_and_align_labels,
)

rg.init(
    api_url="https://argilla.openfoodfacts.org",
    api_key=os.environ["ARGILLA_API_KEY"],
    workspace="ingredient-detection-ner",
)


def main(
    run_name: str,
    model_name: str = "xlm-roberta-large",
    dataset_version: str = "v1.1-alpha.1",
    num_train_epochs: int = 20,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 64,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    cross_validation_num: int = 5,
):
    os.environ["WANDB_TAGS"] = f"{dataset_version},cross-validation"
    os.environ["WANDB_PROJECT"] = "ingredient-detection-ner"
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "false"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    PARAMS = {
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }

    DATASET_URLS = {
        "train": f"https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-{dataset_version}_train.jsonl.gz",
        "test": f"https://static.openfoodfacts.org/data/datasets/ingredient_detection_dataset-{dataset_version}_test.jsonl.gz",
    }
    base_ds = load_dataset("json", data_files=DATASET_URLS)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Update tokenizer pre-tokenizer: the dataset has been pre-tokenized with Sequence([WhitespaceSplit(), Punctuation()]
    # XLM-Roberta uses a SentencePiece tokenizer with Sequence([WhitespaceSplit(), Metaspace()] as pre-tokenizer
    # Thus we need to add Punctuation() pre-tokenizer to the pipeline to avoid having a train/inference mismatch.
    tokenizer._tokenizer.pre_tokenizer = Sequence(
        [WhitespaceSplit(), Punctuation(), Metaspace()]
    )
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=8
    )

    train_ds = base_ds["train"]
    train_min_num_samples = len(train_ds) // cross_validation_num
    for cross_validation_idx in range(cross_validation_num):
        start = cross_validation_idx * train_min_num_samples
        stop = (
            (cross_validation_idx + 1) * train_min_num_samples
            if cross_validation_idx < cross_validation_num - 1
            else len(train_ds)
        )
        print(start, stop)

        base_cv_test_ds = train_ds.select(range(start, stop))
        base_cv_train_ds = train_ds.select(
            indices=[
                idx
                for idx in range(len(train_ds))
                if idx not in set(range(start, stop))
            ]
        )
        base_cv_ds = DatasetDict({"train": base_cv_train_ds, "test": base_cv_test_ds})
        cv_ds = base_cv_ds.map(
            functools.partial(tokenize_and_align_labels, tokenizer=tokenizer),
            batched=True,
        )

        # Instanciate a new model for each cross-validation fold
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=3, id2label=id2label, label2id=label2id
        )
        cv_run_name = f"{run_name}-cv-{cross_validation_idx}"

        training_args = TrainingArguments(
            output_dir=f"ingredient-detection-{model_name}",
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_f1",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to=None,
            fp16=fp16,
            save_total_limit=10,
            run_name=cv_run_name,
            **PARAMS,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=cv_ds["train"],
            eval_dataset=cv_ds["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        if num_train_epochs > 0:
            trainer.train()

        save_prediction_artifacts(
            cv_run_name,
            model,
            tokenizer,
            cv_ds,
            per_device_eval_batch_size,
            argilla_ds_name=f"{cv_run_name}_predictions",
            # No need to log artifacts to wandb for cross-validation
            log_wandb=False,
        )
        # Delete model to free up memory
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    typer.run(main)
