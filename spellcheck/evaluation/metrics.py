from typing import List

import pandas as pd
from statistics import mean
from difflib import SequenceMatcher
from ingredients import format_ingredients, normalize_ingredients


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
        output["ingr_precision"] = not_failing_mean(self.items_ingr_precision)
        output["ingr_recall"] = not_failing_mean(self.items_ingr_recall)
        output["ingr_fidelity"] = not_failing_mean(self.items_ingr_fidelity)

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
            per_item_ingredients_metrics(item, prediction_txt)
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
            txt_similarity(item['correct'], prediction_txt)
            for item, prediction_txt
            in zip(self.items, self.prediction_txts)
        ]


def per_item_ingredients_metrics(item, prediction_txt):
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
    original_ingredients = format_ingredients(item["original"])
    correct_ingredients = format_ingredients(item["correct"])
    predicted_ingredients = format_ingredients(prediction_txt)

    return {
        "precision": ratio(
            (correct_ingredients & predicted_ingredients) - original_ingredients,
            predicted_ingredients - original_ingredients,
        ),
        "recall": ratio(
            (correct_ingredients & predicted_ingredients) - original_ingredients,
            correct_ingredients - original_ingredients,
        ),
        "fidelity": ratio(
            original_ingredients & correct_ingredients & predicted_ingredients,
            original_ingredients & correct_ingredients,
        ),
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


def txt_similarity(correct_txt, prediction_txt):
    matcher = SequenceMatcher(is_junk, correct_txt, prediction_txt)
    return 100.0 * matcher.ratio()


def is_junk(c):
    return False
