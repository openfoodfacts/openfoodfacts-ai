import pandas as pd
from typing import List
from statistics import mean
from difflib import SequenceMatcher
from ingredients import (
    normalize_ingredients,
    normalize_item_ingredients,
    tokenize_ingredients,
)


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
            "number_should_have_been_changed": safe_sum(self.items_should_have_changed),
            "number_changed": safe_sum(self.items_changed),
            "number_correct_when_changed": safe_sum(
                self.items_correct_answer_when_changed
            ),
        }

        # Exact text metrics
        output["txt_precision"] = safe_ratio(
            output["number_correct_when_changed"], output["number_changed"]
        )
        output["txt_recall"] = safe_ratio(
            output["number_correct_when_changed"],
            output["number_should_have_been_changed"],
        )
        output["txt_similarity_metric"] = safe_mean(self.items_txt_similarity)

        # Ingredients metrics micro
        output["ingr_micro_precision"] = safe_mean(self.items_ingr_precision)
        output["ingr_micro_recall"] = safe_mean(self.items_ingr_recall)
        output["ingr_micro_fidelity"] = safe_mean(self.items_ingr_fidelity)

        # Ingredients metrics macro
        output["ingr_macro_precision"] = safe_ratio(
            sum(x["precision_num"] for x in self.items_ingr_metrics),
            sum(x["precision_den"] for x in self.items_ingr_metrics),
        )
        output["ingr_macro_recall"] = safe_ratio(
            sum(x["recall_num"] for x in self.items_ingr_metrics),
            sum(x["recall_den"] for x in self.items_ingr_metrics),
        )
        output["ingr_macro_fidelity"] = safe_ratio(
            sum(x["fidelity_num"] for x in self.items_ingr_metrics),
            sum(x["fidelity_den"] for x in self.items_ingr_metrics),
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
    original_ingredients = tokenize_ingredients(original, remove_plural=True)
    correct_ingredients = tokenize_ingredients(correct, remove_plural=True)
    predicted_ingredients = tokenize_ingredients(prediction, remove_plural=True)

    original_count = len(original_ingredients)
    correct_count = len(correct_ingredients)
    predicted_count = len(predicted_ingredients)

    correct_predicted_count = _matching_tokens_count(
        correct_ingredients, predicted_ingredients,
    )
    original_correct_count = _matching_tokens_count(
        original_ingredients, correct_ingredients,
    )
    original_predicted_count = _matching_tokens_count(
        original_ingredients, predicted_ingredients,
    )
    original_correct_predicted_count = _matching_tokens_count(
        original_ingredients, predicted_ingredients, correct_ingredients,
    )

    precision_num = correct_predicted_count - original_correct_predicted_count
    precision_den = predicted_count - original_predicted_count

    recall_num = correct_predicted_count - original_correct_predicted_count
    recall_den = correct_count - original_correct_count

    fidelity_num = original_correct_predicted_count
    fidelity_den = original_correct_count

    return {
        "precision_num": precision_num,
        "precision_den": precision_den,
        "precision": safe_ratio(precision_num, precision_den),
        "recall_num": recall_num,
        "recall_den": recall_den,
        "recall": safe_ratio(recall_num, recall_den),
        "fidelity_num": fidelity_num,
        "fidelity_den": fidelity_den,
        "fidelity": safe_ratio(fidelity_num, fidelity_den),
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    if isinstance(numerator, set):
        numerator = len(numerator)
    if isinstance(denominator, set):
        denominator = len(denominator)
    if denominator == 0:
        return
    return 100.0 * numerator / denominator


def safe_mean(l):
    l = [item for item in l if item is not None]
    if not l:
        return
    return mean(l)


def safe_sum(l):
    return sum([item for item in l if item is not None])


def safe_matching_tokens_count(a: List[str], b: List[str]) -> int:
    count_a = _matching_tokens_count(a, b)
    count_b = _matching_tokens_count(b, a)
    if count_a != count_b:
        print("Matching tokens count is not symmetric !!!")
    return (count_a + count_b) / 2
    if count_a != count_b:
        raise ValueError("Matching tokens count is not symmetric !")
    return count_a


def _matching_tokens_count(*args: List[str]) -> List[str]:
    return len(_matching_tokens(*args))


def _matching_tokens(*args: List[str]) -> List[str]:
    n_args = len(args)
    tokens_list = []
    tokens_list_a = []
    tokens_list_b = []
    if n_args == 2:
        tokens_list_a = _pair_matching_tokens(args[0], args[1])
        tokens_list_b = _pair_matching_tokens(args[1], args[0])
    else:
        for i_arg in range(n_args):
            arg = args[i_arg]
            args_without_i = args[:i_arg] + args[i_arg + 1 :]
            tokens_list_a = _pair_matching_tokens(
                arg, _matching_tokens(*args_without_i)
            )
            tokens_list_b = _pair_matching_tokens(
                _matching_tokens(*args_without_i), arg
            )
    if len(tokens_list_a) > len(tokens_list):
        tokens_list = tokens_list_a
    if len(tokens_list_b) > len(tokens_list):
        tokens_list = tokens_list_b
    return tokens_list


def _pair_matching_tokens(a: List[str], b: List[str]) -> List[str]:
    matching_blocks = SequenceMatcher(is_junk_token, a, b).get_matching_blocks()
    matching_tokens = []
    for block in matching_blocks:
        matching_tokens += a[block.a : block.a + block.size]
    return matching_tokens


def txt_similarity(txt_a: str, txt_b: str) -> float:
    matcher = SequenceMatcher(None, txt_a, txt_b)
    return 100.0 * matcher.ratio()


def is_junk_token(token: str) -> bool:
    return token in {" "}


def _pair_matching_tokens(a: List[str], b: List[str]) -> List[str]:
    matching_blocks = SequenceMatcher(is_junk_token, a, b).get_matching_blocks()
    matching_tokens = []
    for block in matching_blocks:
        matching_tokens += a[block.a : block.a + block.size]
    return matching_tokens
