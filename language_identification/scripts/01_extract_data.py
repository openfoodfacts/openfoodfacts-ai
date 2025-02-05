import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset


def main():
    # extracting all texts with their languages from huggingface dataset

    # path where to save selected data
    dataset_file = os.path.join(Path(__file__).parent, "texts_with_lang.csv")

    hf_dataset = load_dataset("openfoodfacts/product-database", split="food")

    data = set()
    for entry, main_lang in zip(hf_dataset["ingredients_text"], hf_dataset["lang"]):  # iterate over products
        for product_in_lang in entry:
            if product_in_lang["text"]:
                lang = main_lang if product_in_lang["lang"] == "main" else product_in_lang["lang"]
                data.add((product_in_lang["text"], lang))

    df = pd.DataFrame(data, columns=["ingredients_text", "lang"])
    df.dropna(inplace=True)
    df.to_csv(dataset_file, index=False)


if __name__ == "__main__":
    main()
