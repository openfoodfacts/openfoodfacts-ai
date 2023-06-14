import functools
import html
import os
from pathlib import Path

import evaluate
import numpy as np
import orjson
import seqeval
import typer
import wandb
from datasets import load_dataset
from tokenizers.pre_tokenizers import Metaspace, Punctuation, Sequence, WhitespaceSplit
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

id2label = {
    0: "O",
    1: "B-ING",
    2: "I-ING",
}
label2id = {v: k for k, v in id2label.items()}
label_list = list(id2label.values())


def convert_pipeline_output_to_html(text: str, output: list[dict]):
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
    model, tokenizer, dataset, per_device_eval_batch_size: int
):
    classifier = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        aggregation_strategy="simple",
    )
    no_aggregation_classifier = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        aggregation_strategy=None,
    )
    artifact = wandb.Artifact(f"predictions", type="prediction")

    for split_name in ("test", "train"):
        texts = dataset[split_name]["text"]
        outputs = classifier(texts, batch_size=per_device_eval_batch_size)
        html_items = ["<html>\n<body>"]
        for text, output in zip(texts, outputs):
            html_item = convert_pipeline_output_to_html(text, output)
            html_items.append(html_item)

        html_items.extend(["</body>\n</html>"])
        html_str = "\n".join(html_items)
        prediction_html_file_path = Path(f"{split_name}_predictions.html")
        prediction_html_file_path.write_text(html_str)
        artifact.add_file(prediction_html_file_path)
        no_aggregated_output = no_aggregation_classifier(
            texts, batch_size=per_device_eval_batch_size
        )
        prediction_file_path = Path(f"{split_name}_predictions.jsonl")
        prediction_file_path.write_text(
            "\n".join(
                orjson.dumps(x, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
                for x in no_aggregated_output
            )
        )
        artifact.add_file(prediction_file_path)

    wandb.log_artifact(artifact)


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
    tokens: list[str], labels: list[int], id2label: dict[int, str]
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
    dataset_version: str = "v4",
    num_train_epochs: int = 20,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 64,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
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
        report_to="wandb",
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

    trainer.train()
    save_prediction_artifacts(model, tokenizer, ds, per_device_eval_batch_size)


if __name__ == "__main__":
    typer.run(main)
