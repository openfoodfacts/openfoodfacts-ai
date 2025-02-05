import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import trange
from typing import Sequence

from sklearn.metrics import confusion_matrix

from language_identification.scripts.inference import calculate_fasttext_predictions, calculate_lingua_predictions


def replace_lang_code(model_predictions: list[str], mapping: dict[str, str]) -> None:
    """
    Replace predicted language codes in the `model_predictions` list
    using `mapping`, where:
      - Keys represent the original language codes (predicted by the model)
      - Values represent the target language codes to replace them with.

    The purpose of this function is to standardize language codes
    by combining multiple variants of the same language into a unified code
    in order to match supported languages.
    """
    for i in trange(len(model_predictions)):
        if model_predictions[i] in mapping:
            model_predictions[i] = mapping[model_predictions[i]]


def calculate_metrics(cm: np.ndarray, labels: Sequence, model_name: str) -> pd.DataFrame:
    """
    Calculate precision, recall and f1-score.
    Args:
        cm: confusion matrix
        labels: languages, for which the metrics need to be calculated
        model_name: model name (needed for column names in DataFrame)

    Returns: pandas.DataFrame with computed metrics for each language.
    """
    tp_and_fn = cm.sum(axis=1)
    tp_and_fp = cm.sum(axis=0)
    tp = cm.diagonal()

    precision = np.divide(tp, tp_and_fp, out=np.zeros_like(tp, dtype=float), where=tp_and_fp > 0)
    recall = np.divide(tp, tp_and_fn, out=np.zeros_like(tp, dtype=float), where=tp_and_fn > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float),
                   where=(precision + recall) > 0)

    df = pd.DataFrame({
        "lang": labels,
        f"{model_name}_precision": precision,
        f"{model_name}_recall": recall,
        f"{model_name}_f1": f1,
    })

    return df


def main():
    texts_under_10_words = pd.read_csv(os.path.join(Path(__file__).parent, "texts_under_10_words.csv"))
    texts = texts_under_10_words.ingredients_text.tolist()
    true_labels = texts_under_10_words.lang.tolist()
    possible_class_labels = texts_under_10_words["lang"].value_counts().index.tolist()  # use value_counts in order to get sorted by frequency list

    fasttext_preds = calculate_fasttext_predictions(texts)
    lingua_preds = calculate_lingua_predictions(texts)

    mapping = {"yue": "zh"}  # yue is a type of Chinese
    replace_lang_code(fasttext_preds, mapping)
    replace_lang_code(lingua_preds, mapping)

    predictions = [fasttext_preds, lingua_preds]
    model_names = ["fasttext", "lingua"]
    metrics = []
    for preds, model_name in zip(predictions, model_names):
        cm = confusion_matrix(true_labels, preds, labels=possible_class_labels)
        cm_df = pd.DataFrame(cm, index=possible_class_labels, columns=possible_class_labels)
        cm_df.to_csv(os.path.join(Path(__file__).parent, f"{model_name}_confusion_matrix.csv"))

        metrics_df = calculate_metrics(cm, possible_class_labels, model_name)
        metrics_df.set_index("lang", inplace=True)
        metrics.append(metrics_df)

    # combine results
    metrics_df = pd.DataFrame(texts_under_10_words.lang.value_counts())
    metrics_df = pd.concat((metrics_df, *metrics), axis=1)

    # change columns order
    metrics_df = metrics_df[
        ["count"] + [f"{model}_{metric}" for metric in ["precision", "recall", "f1"] for model in model_names]
    ]

    metrics_df.to_csv(os.path.join(Path(__file__).parent, "10_words_metrics.csv"))

if __name__ == "__main__":
    main()
