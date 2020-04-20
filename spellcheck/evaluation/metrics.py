from difflib import SequenceMatcher
from statistics import mean
from typing import List

import pandas as pd
from ingredients import normalize_ingredients, tokenize_ingredients


def normalize_item_ingredients(item):
    item = item.copy()
    item["original"] = normalize_ingredients(item["original"])
    item["correct"] = normalize_ingredients(item["correct"])
    return item


class Evaluation(object):
    """docstring for Evaluation."""

    def __init__(self, items, prediction_txts):
        self.items = [normalize_item_ingredients(i) for i in items]
        self.prediction_txts = [normalize_ingredients(t) for t in prediction_txts]

        self.items_should_have_changed = None
        self.items_changed = None
        self.items_correct_answer_when_changed = None
        self.items_ingr_metrics = None
        self.items_ingr_precision = None
        self.items_ingr_recall = None
        self.items_ingr_fidelity = None
        self.items_txt_similarity = None
        self._preprocess()

    def metrics(self):
        # Items
        output = {
            "number_items": len(self.items),
            "number_should_have_been_changed": not_failing_sum(
                self.items_should_have_changed
            ),
            "number_changed": not_failing_sum(self.items_changed),
            "number_correct_when_changed": not_failing_sum(
                self.items_correct_answer_when_changed
            ),
        }

        # Exact text metrics
        output["txt_precision"] = ratio(
            output["number_correct_when_changed"], output["number_changed"]
        )
        output["txt_recall"] = ratio(
            output["number_correct_when_changed"],
            output["number_should_have_been_changed"],
        )
        output["txt_similarity_metric"] = not_failing_mean(self.items_txt_similarity)

        # Ingredients metrics
        output["ingr_micro_precision"] = not_failing_mean(self.items_ingr_precision)
        output["ingr_micro_recall"] = not_failing_mean(self.items_ingr_recall)
        output["ingr_micro_fidelity"] = not_failing_mean(self.items_ingr_fidelity)
        output["ingr_macro_precision"] = (
            sum(x["precision_num"] for x in self.items_ingr_metrics)
            * 100
            / sum(x["precision_den"] for x in self.items_ingr_metrics)
        )
        output["ingr_macro_recall"] = (
            sum(x["recall_num"] for x in self.items_ingr_metrics)
            * 100
            / sum(x["recall_den"] for x in self.items_ingr_metrics)
        )

        return output

    def detailed_dataframe(self):
        return pd.DataFrame(
            list(
                zip(
                    [item["_id"] for item in self.items],
                    self.items_should_have_changed,
                    self.items_changed,
                    self.items_correct_answer_when_changed,
                    self.items_ingr_precision,
                    self.items_ingr_recall,
                    self.items_ingr_fidelity,
                    self.items_txt_similarity,
                )
            ),
            columns=[
                "_id",
                "should_have_changed",
                "changed",
                "correct_answer_when_changed",
                "ingr_precision",
                "ingr_recall",
                "ingr_fidelity",
                "txt_similarity",
            ],
        )

    def _preprocess(self):
        self.items_should_have_changed = [
            item["original"] != item["correct"] for item in self.items
        ]

        self.items_changed = [
            item["original"] != prediction_txt
            for item, prediction_txt in zip(self.items, self.prediction_txts)
        ]

        self.items_correct_answer_when_changed = [
            item["correct"] == prediction_txt if item_changed else None
            for item, prediction_txt, item_changed in zip(
                self.items, self.prediction_txts, self.items_changed
            )
        ]

        self.items_ingr_metrics = [
            per_item_ingredients_metrics(
                item["original"], item["correct"], prediction_txt
            )
            for item, prediction_txt in zip(self.items, self.prediction_txts)
        ]
        self.items_ingr_precision = [
            metric["precision"] for metric in self.items_ingr_metrics
        ]
        self.items_ingr_recall = [
            metric["recall"] for metric in self.items_ingr_metrics
        ]
        self.items_ingr_fidelity = [
            metric["fidelity"] for metric in self.items_ingr_metrics
        ]

        self.items_txt_similarity = [
            txt_similarity(item["correct"], prediction_txt)
            for item, prediction_txt in zip(self.items, self.prediction_txts)
        ]


def per_item_ingredients_metrics(original: str, correct: str, prediction: str):
    """
    Precision :
        number of times a change introduce a correct ingredient
            / number of time we change an ingredient

    Recall :
        number of times a change introduce a correct ingredient
            / number of time we should have changed an ingredient

    Fidelity :
        number of time we did not change an ingredient when it was already correct
            / number of time we changed an ingredient that was correct but isn't anymore
    """
    original_ingredients = tokenize_ingredients(original)
    correct_ingredients = tokenize_ingredients(correct)
    predicted_ingredients = tokenize_ingredients(prediction)
    original_count = len(original_ingredients)
    correct_count = len(correct_ingredients)
    predicted_count = len(predicted_ingredients)
    min_original_correct_count = min(original_count, correct_count)
    correct_predicted_count = get_matching_token_count(
        correct_ingredients, predicted_ingredients
    )
    original_correct_count = get_matching_token_count(
        original_ingredients, correct_ingredients
    )
    original_predicted_count = get_matching_token_count(
        original_ingredients, predicted_ingredients
    )
    print(f"original_count: {original_count}")
    print(f"correct_count: {correct_count}")
    print(f"predicted_count: {predicted_count}")
    print(f"correct_predicted_count: {correct_predicted_count}")
    print(f"original_correct_count: {original_correct_count}")

    precision_num = correct_predicted_count - original_correct_count
    recall_num = precision_num
    precision_den = predicted_count - original_predicted_count
    recall_den = min_original_correct_count - original_correct_count

    return {
        "precision": ratio(precision_num, precision_den),
        "recall": ratio(recall_num, recall_den),
        "fidelity": 100,
        "precision_num": precision_num,
        "precision_den": precision_den,
        "recall_num": recall_num,
        "recall_den": recall_den,
    }


def ratio(numerator, denominator):
    if isinstance(numerator, set):
        numerator = len(numerator)
    if isinstance(denominator, set):
        denominator = len(denominator)
    if denominator == 0:
        return
    return 100.0 * numerator / denominator


def not_failing_mean(l):
    l = [item for item in l if item is not None]
    if not l:
        return
    return mean(l)


def not_failing_sum(l):
    return sum([item for item in l if item is not None])


def get_matching_token_count(a: List[str], b: List[str]) -> int:
    matching_blocks = SequenceMatcher(is_junk_token, a, b).get_matching_blocks()
    matching_blocks = matching_blocks[:-1]
    return sum(x.size for x in matching_blocks)


def txt_similarity(correct: str, prediction: str):
    matcher = SequenceMatcher(None, correct, prediction)
    return 100.0 * matcher.ratio()


def is_junk_token(token: str) -> bool:
    return token in {" "}
