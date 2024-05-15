from typing import List, Mapping, Tuple

import polars as pl

from spellcheck.utils import get_repo_dir, get_logger


REPO_DIR = get_repo_dir()
DATA_PATH = REPO_DIR / "data/database/openfoodfacts-products.jsonl"
OUTPUT_DATA_PATH = REPO_DIR / "data/dataset/0_extracted_lists_of_ingredients.parquet"

FEATURE_NAMES = [
    "code",
    "lang",
    "ingredients_text",
    "unknown_ingredients_n",
    "known_ingredients_n",
    "ingredients_n"
]
DTYPES_MAPPING = {
    "ingredients_n": pl.Int16,
    "unknown_ingredients_n": pl.Int16,
}
DATASET_SIZE = 3000
PERCENTAGE_UNKNOWN_RANGE = (0.2, 0.4)  # Min, Max desired
SEED = 42

LOGGER = get_logger()


def main():

    if not DATA_PATH.is_file():
        raise ValueError(f"Data path is not valid: {str(DATA_PATH)}")
    
    lazy_df = pl.scan_ndjson(DATA_PATH)
    df = extract_data(
        df=lazy_df,
        keep_features=FEATURE_NAMES,
        dtype_output_mapping=DTYPES_MAPPING,
        dataset_size=DATASET_SIZE,
        percentage_unknown_range=PERCENTAGE_UNKNOWN_RANGE,
        seed=SEED
    )
    LOGGER.info(f"The extracted dataset contains {len(df)} rows.")
    df.write_parquet(OUTPUT_DATA_PATH)


def extract_data(    
    df: pl.LazyFrame,
    percentage_unknown_range: Tuple[float, float],
    keep_features: List[str],
    dtype_output_mapping: Mapping,
    dataset_size: int,
    seed: int
) -> pl.DataFrame:
    """Extracts products from the JSONL database based on their percentage unknown range.

    Notes: 
        Take around 8 minutes to run.

    Args:
        df (pl.LazyFrame): Polars Lazyframe (https://docs.pola.rs/py-polars/html/reference/lazyframe)
        percentage_unknown_range (Tuple[float, float]): Tuple specifying the minimum and maximum desired percentage
            of unknown ingredients in the filtered dataset.
        keep_features (List[str]): List of feature names to keep in the output DataFrame.
        dtype_output_mapping (Mapping): Mapping of feature names to their desired data types.
        dataset_size (int): Desired size of the output dataset.
        seed (int): Random seed . Default to 42.

    Returns:
        pl.DataFrame: Polars dataframe
    """
    percentage_min, percentage_max = percentage_unknown_range
    output_df = (
        df.select(pl.col(*keep_features))
        .drop_nulls()
        .with_columns((pl.col("unknown_ingredients_n") / pl.col("ingredients_n")).alias("fraction"))
        .filter((pl.col("fraction") >= percentage_min) & (pl.col("fraction") <= percentage_max))
        .collect(streaming=True)
        .sample(n=dataset_size, shuffle=True, seed=seed)
        .cast(dtype_output_mapping)
    )
    return output_df


if __name__ == "__main__":
    main()
