"""This script is used to compare model predictions with the ground truth
labels.

It loads the dataset and processes each sample with the model. It then
compares the
predicted labels with the ground truth labels and logs the errors in a TSV
file.

This is useful to spot possible annotation errors in the dataset, in both
splits.
"""

import csv
import logging

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from openfoodfacts.types import JSONType
from transformers import AutoModelForTokenClassification, AutoProcessor, BatchEncoding

logger = logging.getLogger(__name__)


def gather_pre_entities(
    logits: torch.Tensor,
    words: list[str],
    batch_encoding: BatchEncoding,
    id2label: dict[int, str],
) -> list[JSONType]:
    """Gather the pre-entities extracted by the model.

    This function takes as input the predicted logits returned by the model and
    additional preprocessing outputs (words, char_offsets, batch_encoding) and returns a
    list of pre-entities with the following fields:

    - `word`: the word corresponding to the entity (string)
    - `entity`: the entity type (string, ex: "ENERGY_KCAL_100G")
    - `score`: the score of the entity (float)
    - `index`: the index of the word in the input
    - `char_start`: the character start index of the entity
    - `char_end`: the character end index of the entity

    :param logits: the predicted logits
    :param words: the words corresponding to the input
    :param char_offsets: the character offsets of the words
    :param batch_encoding: the BatchEncoding returned by the tokenizer
    :param id2label: a dictionary mapping label IDs to label names
    :return: a list of pre-entities
    """
    special_tokens_mask = batch_encoding.special_tokens_mask.numpy()[0]
    logits = logits.numpy()

    maxes = np.max(logits, axis=-1, keepdims=True)
    shifted_exp = np.exp(logits - maxes)
    scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
    label_ids = logits.argmax(axis=-1)

    pre_entities = []
    previous_word_id = None
    word_ids = batch_encoding.word_ids()

    for idx in range(len(scores)):
        # idx may be out of bounds if the input_ids are padded
        # word_id corresponds to the index of the input words, while
        # idx is the index of the token. A word can have multiple tokens
        # if it is split into subwords.
        word_id = word_ids[idx] if idx < len(word_ids) else None
        # Filter special_tokens (BOS, EOS, PAD)
        if special_tokens_mask[idx]:
            previous_word_id = word_id
            continue

        # The token is a subword if it has the same word_id as the previous token
        is_subword = word_id == previous_word_id
        if int(batch_encoding.input_ids[0, idx]) == 3:  # unknown token
            is_subword = False

        if is_subword:
            # If the token is a subword, we skip it
            # The entity will be attached to the first token of the word
            # and the score will be the score of the first token
            continue

        previous_word_id = word_id
        word = words[word_id]
        label_id = label_ids[idx]
        score = float(scores[idx, label_id])
        label = id2label[label_id]
        # As the entities are very short (< 3 tokens most of the time) and as
        # two entities with the same label are in practice never adjacent,
        # we simplify the schema by ignoring the B- and I- prefix.
        # It simplifies processing and makes it more robust against model
        # prefix mis-predictions.
        entity = label.split("-", maxsplit=1)[-1]

        pre_entity = {
            "word": word,
            "entity": entity,
            "label_id": label_id,
            "score": score,
            "index": word_id,
        }
        pre_entities.append(pre_entity)
    return pre_entities


model_name_or_path = "openfoodfacts/nutrition-extractor"
processor = AutoProcessor.from_pretrained(model_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

split_name = "train"
ds = load_dataset("openfoodfacts/nutrient-detection-layout")
val_ds = ds[split_name]
id2label = model.config.id2label
start_at_idx = 0


with open(f"errors_{split_name}.tsv", "w", newline="") as f:
    csv_writer = csv.DictWriter(
        f, fieldnames=["barcode", "image_url", "label_studio_url", "comment"]
    )
    csv_writer.writeheader()

    with torch.inference_mode():
        for i in tqdm.tqdm(range(len(val_ds)), desc="samples"):
            sample = val_ds[i]
            images = sample["image"]
            words = sample["tokens"]
            boxes = sample["bboxes"]
            ner_tags = sample["ner_tags"]
            meta = sample["meta"]
            barcode = meta["barcode"]
            image_url = meta["image_url"]
            task_id = meta["label_studio_id"]
            print(f"Processing sample {i}, barcode: {barcode} - image_url: {image_url}")

            if i < start_at_idx:
                continue

            encoding = processor(
                images,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            # Logits is a tensor of shape
            # (sequence_length, num_labels)
            logits = model(
                **{
                    k: v
                    for k, v in encoding.items()
                    if k not in ("special_tokens_mask", "offset_mapping")
                }
            ).logits[0]
            pre_entities = gather_pre_entities(
                logits,
                words,
                batch_encoding=encoding,
                id2label=id2label,
            )

            if len(pre_entities) != len(ner_tags):
                if logits.shape[0] == 512:
                    print("Warning: the input is too long, it has been truncated")
                else:
                    print(
                        f"Error: different number of entities detected ({len(pre_entities)}) and expected ({len(ner_tags)})"
                    )
                continue

            predicted_ner_tags = [pre_entity["label_id"] for pre_entity in pre_entities]

            error_message = ""
            for i in range(len(predicted_ner_tags)):
                predicted_ner_tag = predicted_ner_tags[i]
                ner_tag = ner_tags[i]
                word = pre_entities[i]["word"].strip()
                if predicted_ner_tag != ner_tag:
                    error_message += f"word: '{word}', predicted: {id2label[predicted_ner_tag]}, expected: {id2label[ner_tag]}|"

            error_message.strip().strip("\n").strip("|")
            if error_message:
                csv_writer.writerow(
                    {
                        "barcode": barcode,
                        "image_url": image_url,
                        "label_studio_url": f"https://annotate.openfoodfacts.org/projects/42/data?tab=68&task={task_id}",
                        "comment": error_message,
                    }
                )
                f.flush()
