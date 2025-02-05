from typing import Sequence

import langcodes
import fasttext
from lingua import LanguageDetectorBuilder
from huggingface_hub import hf_hub_download


def calculate_fasttext_predictions(texts: list[str]) -> list[str]:
    """
    Detect the languages of the given texts using FastText model.

    Args:
        texts: A sequence of texts for which the language detection
               needs to be performed.

    Returns:
        A list of predicted ISO 639-1 language codes.
    """
    # texts preprocessing
    table = str.maketrans({"\n": " ", "\r": " "})
    texts = [text.translate(table).lower() for text in texts]

    # load model
    fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    fasttext_model = fasttext.load_model(fasttext_model_path)

    # get predictions
    predictions, _ = fasttext_model.predict(texts)
    predicted_lang = [predictions[i][0].removeprefix("__label__") for i in range(len(texts))]
    # get iso 639-3
    fasttext_preds = [predicted_lang[i].split("_")[0] for i in range(len(predicted_lang))]
    # get iso 639-1
    fasttext_preds = [langcodes.standardize_tag(lang_code, macro=True) for lang_code in fasttext_preds]
    return fasttext_preds


def calculate_lingua_predictions(texts: Sequence[str]) -> list[str]:
    """
    Detect the languages of the given texts using the Lingua language detector.

    Args:
        texts: A sequence of texts for which the language detection
               needs to be performed.

    Returns:
        A list of predicted ISO 639-1 language codes.
        If the model is uncertain about the language of a text, "xx" is returned
        as a placeholder.
    """
    texts = [text.lower() for text in texts]
    lingua_detector = LanguageDetectorBuilder.from_all_languages().build()

    predictions = lingua_detector.detect_languages_in_parallel_of(texts)

    lingua_preds = [prediction.iso_code_639_1.name.lower() if prediction is not None else "xx"
                    for prediction in predictions]
    return lingua_preds
