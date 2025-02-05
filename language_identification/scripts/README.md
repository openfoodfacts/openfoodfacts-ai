# Research of the quality of models on ingredient texts up to 10 words long.

`01_extract_data.py`: extracts all texts with their languages from [huggingface dataset](https://huggingface.co/datasets/openfoodfacts/product-database).

`02_select_short_texts_with_known_ingredients.py`: filters texts with length up to 10 words, performs ingredient analysis by OFF API, selects ingredient texts with at least 80% of known ingredients, adds short texts from manually checked data.

What is manually checked data: \
I created a validation dataset from texts from OFF (42 languages, 15-30 texts per language).
I took 30 random texts in each language, obtained language predictions using the Deepl API and two other models ([language-detection-fine-tuned-on-xlm-roberta-base](https://huggingface.co/ivanlau/language-detection-fine-tuned-on-xlm-roberta-base) and [multilingual-e5-language-detection](https://huggingface.co/Mike0307/multilingual-e5-language-detection)). For languages they don’t support, I used Google Translate and ChatGPT for verification. (As a result, after correcting the labels, some languages have fewer than 30 texts).


`03_calculate_metrics.py`: obtains predictions by [FastText](https://huggingface.co/facebook/fasttext-language-identification) and [lingua language detector](https://github.com/pemistahl/lingua-py) models for texts up to 10 words long, and calculates precision, recall and f1-score.

Results are in files: [metrics](./10_words_metrics.csv), [FastText confusion matrix](./fasttext_confusion_matrix.csv), [lingua confusion matrix](./lingua_confusion_matrix.csv).

It turned out that both models demonstrate low precision and high recall for some languages (indicating that the threshold might be too high and should be adjusted).