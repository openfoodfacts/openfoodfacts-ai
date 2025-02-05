import os
from pathlib import Path
import unicodedata
import pandas as pd
from tqdm import tqdm
from openfoodfacts import API, APIVersion, Environment
from polyglot.text import Text


def is_punctuation(word: str) -> bool:
    """
    Check if the string `word` is a punctuation mark.
    """
    return all(unicodedata.category(char).startswith("P") for char in word)


def select_texts_by_len(data: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    """
    Select rows from `data` where the number of words in `ingredients_text` 
    is between `min_len` and `max_len` inclusively, excluding punctuation.

    Args:
        data: pandas.DataFrame with columns `ingredients_text`, `lang`
        min_len: Minimum number of words.
        max_len: Maximum number of words.

    Returns:
        A pandas DataFrame containing the rows from the `data` that satisfy the word count condition.
    """
    selected_rows = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        # the object that recognizes individual words in text
        text = Text(row.ingredients_text, hint_language_code=row.lang)
        # `Text` recognizes punctuation marks as words
        words = [word for word in text.words if not is_punctuation(word)]
        if min_len <= len(words) <= max_len:
            selected_rows.append(row)

    selected_df = pd.DataFrame(selected_rows)
    return selected_df


def main():
    dataset_file = os.path.join(Path(__file__).parent, "texts_with_lang.csv")
    all_data = pd.read_csv(dataset_file)

    short_texts_df = select_texts_by_len(all_data, min_len=0, max_len=10)

    # perform ingredient analysis
    api = API(user_agent="langid",
              version=APIVersion.v3,
              environment=Environment.net)

    threshold = 0.8
    texts_with_known_ingredients = []

    # select ingredients texts with the rate of known ingredients >= `threshold`
    for i, row in tqdm(short_texts_df.iterrows(), total=len(short_texts_df)):
        try:
            ingredient_analysis_results = api.product.parse_ingredients(row.ingredients_text, lang=row.lang)
        except RuntimeError:
            continue

        is_in_taxonomy = sum(dct.get("is_in_taxonomy", 0) for dct in ingredient_analysis_results)
        is_in_taxonomy_rate = is_in_taxonomy / len(ingredient_analysis_results) \
            if len(ingredient_analysis_results) > 0 else 0.

        if is_in_taxonomy_rate >= threshold:
            texts_with_known_ingredients.append(row)

    texts_with_known_ingredients_df = pd.DataFrame(texts_with_known_ingredients)

    # add short texts from manually checked data
    manually_checked_data = pd.read_csv(os.path.join(Path(__file__).parent, "manually_checked_data.csv"))
    short_texts_manual = select_texts_by_len(manually_checked_data, min_len=0, max_len=10)

    # combine data and save
    all_texts_under_10_words = pd.concat((short_texts_manual, texts_with_known_ingredients_df), ignore_index=True)
    all_texts_under_10_words.drop_duplicates(inplace=True, ignore_index=True)
    all_texts_under_10_words.to_csv(os.path.join(Path(__file__).parent, "texts_under_10_words.csv"), index=False)


if __name__ == "__main__":
    main()
