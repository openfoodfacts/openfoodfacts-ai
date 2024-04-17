import os
import logging
from typing import List, Mapping, Tuple
from pathlib import Path

import pandas as pd


REPO_DIR = Path(os.path.dirname(__file__)).parent.parent
DATA_PATH = REPO_DIR / "data/database/openfoodfacts-products.jsonl"
OUTPUT_DATA_PATH = REPO_DIR / "data/dataset/extracted_lists_of_ingredients.parquet"

FEATURE_NAMES = [
    "code",
    "lang",
    "ingredients_text",
    "unknown_ingredients_n",
    "known_ingredients_n"
]
DTYPES_MAPPING = {
    "code": int,
    "lang": str,
    "ingredients_text": str,
    "unknown_ingredients_n": int,
    "known_ingredients_n": int,
}
CHUNKSIZE = 3000
DATASET_SIZE = 5000
PERCENTAGE_UNKNOWN = (0.1, 0.3)  # Min, Max desired

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    df = extract_data(
        data_path=DATA_PATH,
        keep_features=FEATURE_NAMES,
        dtype_output_mapping=DTYPES_MAPPING,
        chunksize=CHUNKSIZE,
        dataset_size=DATASET_SIZE,
        percentage_unknown=PERCENTAGE_UNKNOWN,
    )
    LOGGER.info(f"The extracted dataset contains {len(df)} rows.")
    df.to_parquet(OUTPUT_DATA_PATH)


def extract_data(
    data_path: Path,
    keep_features: List[str],
    dtype_output_mapping: Mapping,
    chunksize: int,
    dataset_size: int,
    percentage_unknown: Tuple[float, float],
) -> pd.DataFrame:
    """Extracts and filters data from a JSONL file.

    Args:
        data_path (Path): Path to the JSONL file containing the data from OFF.
        keep_features (List[str]): List of feature names to keep in the output DataFrame.
        dtype_output_mapping (Mapping): Mapping of feature names to their desired data types.
        chunksize (int): Size of chunks to read from the JSONL file.
        dataset_size (int): Desired size of the output dataset.
        percentage_unknown (Tuple[float, float]): Tuple specifying the minimum and maximum desired percentage
            of unknown ingredients in the filtered dataset.

    Returns:
        pd.DataFrame: Filtered DataFrame containing the extracted data with specified features.

    Raises:
        ValueError: If the provided data path is not a valid file.
    """
    if not data_path.is_file():
        raise ValueError(f"Data path is not valid: {str(data_path)}")

    # Init
    output_df = pd.DataFrame()
    n_rows = 0

    for chunk_df in pd.read_json(data_path, lines=True, chunksize=chunksize):
        # Shuffle data since it is sorted by barcode
        chunk_df = chunk_df.sample(frac=1)

        # DTypes were changed during the process. Fix it to process the data
        chunk_df = chunk_df.dropna(subset=keep_features)
        chunk_df = chunk_df.astype(dtype_output_mapping)

        if n_rows < dataset_size:
            chunk_filtered_df = filter_small_percentage_unknown(
                df=chunk_df, percentage_unknown=percentage_unknown
            )
            output_df = pd.concat(
                [output_df, chunk_filtered_df], axis=0, ignore_index=True
            )
            n_rows += len(chunk_filtered_df)
        else:
            # Truncate to get the desired size (possibility to have less than the desired size)
            output_df = output_df.iloc[:dataset_size]
            break
        LOGGER.info(f"Size of the filtered dataset: {len(output_df)}")

    return output_df[keep_features]


def filter_small_percentage_unknown(
    df: pd.DataFrame, percentage_unknown: Tuple[float, float]
) -> pd.DataFrame:
    """Filters a DataFrame to select rows where the percentage of unrecognized ingredients
    falls within a specified range.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be filtered.
        percentage_unknown (Tuple[float, float]): Tuple specifying the minimum and
            maximum desired percentage of unrecognized ingredients.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    percentage_min, percentage_max = percentage_unknown
    df = df.loc[(df["known_ingredients_n"] > 0) & (df["ingredients_text"] != "")]
    fractions = df["unknown_ingredients_n"] / df["known_ingredients_n"]
    df = df.loc[(percentage_min <= fractions) & (fractions <= percentage_max)]
    return df


if __name__ == "__main__":
    main()
