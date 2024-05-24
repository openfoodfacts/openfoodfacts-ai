"""Flan-T5 training script."""
from typing import Iterable, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, disable_caching
import numpy as np

from spellcheck.utils import get_logger, timer, get_repo_dir
from spellcheck.evaluation.evaluator import SpellcheckEvaluator 


REPO_DIR = get_repo_dir()

# disable_caching()

model_name = "spellcheck"
model_id="google/flan-t5-small"
dataset_id = "openfoodfacts/spellcheck-dataset"
benchmark_id = 'openfoodfacts/spellcheck-benchmark'
padding = "max_length"
max_length = 512
instruction = "Correct the list of ingredients: "
# Hugging Face repository id
repository_id = REPO_DIR / "model-training" / f"{model_id.split('/')[1]}-{model_name}"


def main():

    # Load tokenizer and model of FLAN-t5-base
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Load data
    train_dataset = load_dataset(path=dataset_id, split="train") 
    benchmark_dataset = load_dataset(path=benchmark_id, split="train")

    preprocessed_train_dataset = train_dataset.map(
        preprocess, 
        batched=True, 
        remove_columns=train_dataset.column_names, 
        fn_kwargs={
            "input_name": "text", 
            "target_name": "label",
            "tokenizer": tokenizer,
        }
    )
    preprocessed_benchmark_dataset = benchmark_dataset.map(
        preprocess,
        batched=True,
        remove_columns=benchmark_dataset.column_names,
        fn_kwargs={
            "input_name": "original", 
            "target_name": "reference",
            "tokenizer": tokenizer,
        }
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=0.1,
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
    )
 
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=preprocessed_train_dataset,
        eval_dataset=preprocessed_benchmark_dataset,
    )
    trainer.train()

    
def preprocess(sample, input_name, target_name, tokenizer):
    """Preprocessing step

    Args:
        sample (_type_): _description_

    Returns:
        _type_: _description_
    """
    # add prefix to the input for t5
    inputs = [instruction + item for item in sample[input_name]]
    # tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length, 
        padding=padding, 
        truncation=True
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample[target_name], 
        padding=padding,
        truncation=True
    )
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    

if __name__ == "__main__":
    main()