import functools
import gzip
import html
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import argilla as rg
import evaluate
import numpy as np
import orjson
import seqeval
import typer
import wandb
from datasets import load_dataset
from token_classification_pipeline import (
    TokenClassificationPipeline as CustomTokenClassificationPipeline,
)
from tokenizers.pre_tokenizers import Metaspace, Punctuation, Sequence, WhitespaceSplit
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.pipelines import TokenClassificationPipeline

id2label = {
    0: "O",
    1: "B-ING",
    2: "I-ING",
}
label2id = {v: k for k, v in id2label.items()}
label_list = list(id2label.values())

rg.init(
    api_url="https://argilla.openfoodfacts.org",
    api_key=os.environ["ARGILLA_API_KEY"],
    workspace="ingredient-detection-ner",
)


def convert_pipeline_output_to_html(text: str, output: List[dict]):
    html_str = ""
    previous_idx = 0
    for item in output:
        entity = item["entity_group"]
        score = item["score"]

        if entity != "ING":
            raise ValueError()

        start_idx = item["start"]
        end_idx = item["end"]
        html_str += (
            html.escape(text[previous_idx:start_idx])
            + "<mark>"
            + html.escape(text[start_idx:end_idx])
            + f"</mark>[{score}]"
        )
        previous_idx = end_idx
    html_str += html.escape(text[previous_idx:])
    return f"<p>{html_str}</p>"


def save_prediction_artifacts(
    run_name: str,
    model,
    tokenizer,
    dataset,
    per_device_eval_batch_size: int,
    argilla_ds_name: Optional[str] = None,
    log_wandb: bool = True,
):
    token_classifier_pipeline = CustomTokenClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        aggregation_strategy=None,
    )
    aggregated_token_classifier_pipeline = CustomTokenClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        aggregation_strategy="FIRST",
    )
    artifact = wandb.Artifact(run_name, type="prediction")
    argilla_records = []

    for split_name in ("test", "train"):
        split_ds = dataset[split_name]
        texts = split_ds["text"]
        outputs = token_classifier_pipeline(
            texts, batch_size=per_device_eval_batch_size
        )
        aggregated_outputs = aggregated_token_classifier_pipeline(
            texts, batch_size=per_device_eval_batch_size
        )
        html_items = ["<html>\n<body>"]
        for text, output in zip(texts, aggregated_outputs):
            html_item = convert_pipeline_output_to_html(text, output)
            html_items.append(html_item)

        html_items.extend(["</body>\n</html>"])
        html_str = "\n".join(html_items)
        prediction_html_file_path = Path(f"{split_name}_predictions.html")
        prediction_html_file_path.write_text(html_str)
        artifact.add_file(prediction_html_file_path)
        prediction_file_path = Path(f"{split_name}_predictions.jsonl.gz")
        with gzip.open(prediction_file_path, "wb") as f:
            f.write(
                b"\n".join(
                    orjson.dumps(
                        {
                            "text": split_ds[i]["text"],
                            "meta": split_ds[i]["meta"],
                            "entities": entities,
                        },
                        option=orjson.OPT_SERIALIZE_NUMPY,
                    )
                    for i, entities in enumerate(outputs)
                )
            )
        artifact.add_file(prediction_file_path)

        full_jsonl_output = []
        for i, entities in enumerate(aggregated_outputs):
            sample = split_ds[i]
            text = sample["text"]
            meta = sample["meta"]
            full_jsonl_output.append({"text": text, "meta": meta, "entities": entities})
            try:
                argilla_records.append(
                    rg.TokenClassificationRecord(
                        text=text,
                        tokens=sample["tokens"],
                        annotation_agent="ground-truth",
                        annotation=[
                            ("ING", offsets[0], offsets[1])
                            for offsets in sample["offsets"]
                        ],
                        prediction=[
                            ("ING", entity["start"], entity["end"])
                            for entity in entities
                        ],
                        id=meta["id"],
                        metadata=meta,
                    )
                )
            except ValueError:
                # Argilla perfoms a validation of the predictions with respect to the tokens
                # and may fail if the predictions are not valid.
                # In this case, we log the error and don't log the prediction to Argilla
                argilla_records.append(
                    rg.TokenClassificationRecord(
                        text=text,
                        tokens=sample["tokens"],
                        annotation_agent="ground-truth",
                        annotation=[
                            ("ING", offsets[0], offsets[1])
                            for offsets in sample["offsets"]
                        ],
                        id=meta["id"],
                        metadata={
                            **meta,
                            "error": "Invalid entity group",
                            "offsets": [
                                (entity["start"], entity["end"]) for entity in entities
                            ],
                        },
                    )
                )
        prediction_file_path = Path(f"{split_name}_predictions_agg.jsonl.gz")
        with gzip.open(prediction_file_path, "wb") as f:
            f.write(
                b"\n".join(
                    orjson.dumps(item, option=orjson.OPT_SERIALIZE_NUMPY)
                    for item in full_jsonl_output
                )
            )
        artifact.add_file(prediction_file_path)

    if log_wandb:
        wandb.log_artifact(artifact)

    if not argilla_ds_name:
        argilla_ds_name = f"{run_name}_predictions"

    argilla_ds_name = argilla_ds_name.replace(
        ".", "_"
    )  # Argilla does not allow dots in dataset names
    rg.log(argilla_records, argilla_ds_name)


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def display_labeled_sequence(
    tokens: List[str], labels: List[int], id2label: Dict[int, str]
):
    assert len(tokens) == len(labels)
    output = []
    for token, label in zip(tokens, labels):
        label_name = id2label[label]
        if label_name == "O":
            output.append(token)
        else:
            output.append(f"{token}|{label_name}")
    return " ".join(output)


# flake8: noqa
seqeval = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main(
    run_name: str,
    model_name: str = "xlm-roberta-large",
    dataset_version: str = "alpha-v6",
    num_train_epochs: int = 20,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 64,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    argilla_ds_name: Optional[str] = None,
    log_wandb: bool = True,
):
    os.environ["WANDB_TAGS"] = f"{dataset_version}"
    os.environ["WANDB_PROJECT"] = "ingredient-detection-ner"
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    PARAMS = {
        "num_train_epochs": num_train_epochs,
        # "auto_find_batch_size": True,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "run_name": run_name,
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
    ds = base_ds.map(
        functools.partial(tokenize_and_align_labels, tokenizer=tokenizer), batched=True
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=8
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"ingredient-detection-{model_name}",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb" if log_wandb else None,
        fp16=fp16,
        save_total_limit=10,
        **PARAMS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if num_train_epochs > 0:
        trainer.train()

    save_prediction_artifacts(
        run_name,
        model,
        tokenizer,
        ds,
        per_device_eval_batch_size,
        argilla_ds_name=argilla_ds_name,
        log_wandb=log_wandb,
    )


if __name__ == "__main__":
    typer.run(main)
