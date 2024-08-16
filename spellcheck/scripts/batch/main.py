import argparse
import tempfile
import logging
from typing import List

import pandas as pd
from vllm import LLM, SamplingParams
from google.cloud import storage


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

FEATURES_VALIDATION = ["code", "text"]


def parse() -> argparse.Namespace:
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Spellcheck module.")
    parser.add_argument("--data_bucket", type=str, default="robotoff-spellcheck", help="Bucket name.")
    parser.add_argument("--pre_data_suffix", type=str, default="data/test_data.parquet", help="Dataset suffix containing the data to be processed.")
    parser.add_argument("--post_data_suffix", type=str, default="data/test_processed_data.parquet", help="Dataset suffix containing the processed data.")
    parser.add_argument("--model_path", default="openfoodfacts/spellcheck-mistral-7b", type=str, help="HF model path.")
    parser.add_argument("--max_model_len", default=1024, type=int, help="Maximum model context length. A lower max context length reduces the memory footprint and accelerate the inference.")
    parser.add_argument("--temperature", default=0, type=float, help="Sampling temperature.")
    parser.add_argument("--max_tokens", default=1024, type=int, help="Maximum number of tokens to generate.")
    parser.add_argument("--quantization", default="fp8", type=str, help="Quantization type.")
    parser.add_argument("--dtype", default="auto", type=str, help="Model weights precision. Default corresponds to the modle config (float16 here)")
    return parser.parse_args()


def main():
    """Batch processing job.

    Original lists of ingredients are stored in a gs bucket before being loaded then processed by the model.
    The corrected lists of ingredients are then stored back in gs.

    We use vLLM to process the batch optimaly. The model is loaded from the Open Food Facts Hugging Face model repository.  
    """
    LOGGER.info("Starting batch processing job.")
    args = parse()

    LOGGER.info(f"Loading data from GCS: {args.data_bucket}/{args.pre_data_suffix}")
    data = load_gcs(bucket_name=args.data_bucket, suffix=args.pre_data_suffix)
    LOGGER.info(f"Feature in uploaded data: {data.columns}")
    if not all(feature in data.columns for feature in FEATURES_VALIDATION):
        raise ValueError(f"Data should contain the following features: {FEATURES_VALIDATION}. Current features: {data.columns}")

    instructions = [prepare_instruction(text) for text in data["text"]]
    llm = LLM(
        model=args.model_path, 
        max_model_len=args.max_model_len, 
        dtype=args.dtype,
        quantization=args.quantization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )

    LOGGER.info(f"Starting batch inference:\n {llm}.\n\nSampling parameters: {sampling_params}")
    data["correction"] = batch_inference(instructions, llm=llm, sampling_params=sampling_params)

    LOGGER.info(f"Uploading data to GCS: {args.data_bucket}/{args.post_data_suffix}")
    # Save DataFrame as Parquet to a temporary file
    with tempfile.NamedTemporaryFile(delete=True, suffix='.parquet') as temp_file:
        data.to_parquet(temp_file.name)
        temp_file_name = temp_file.name
        upload_gcs(
            temp_file_name, 
            bucket_name=args.data_bucket, 
            suffix=args.post_data_suffix
        )
    LOGGER.info("Batch processing job completed.")

def prepare_instruction(text: str) -> str:
    """Prepare instruction prompt for fine-tuning and inference.

    Args:
        text (str): List of ingredients

    Returns:
        str: Instruction.
    """
    instruction = (
        "###Correct the list of ingredients:\n"
        + text
        + "\n\n###Correction:\n"
    )
    return instruction


def batch_inference(
        texts: List[str], 
        llm: LLM,
        sampling_params: SamplingParams
    ) -> List[str]:
    """Process batch of texts with vLLM.

    Args:
        texts (List[str]): Batch
        llm (LLM): Model engine optimized with vLLM
        sampling_params (SamplingParams): Generation parameters

    Returns:
        List[str]: Processed batch of texts
    """
    outputs = llm.generate(texts, sampling_params,)
    corrections = [output.outputs[0].text for output in outputs]
    return corrections


def load_gcs(bucket_name: str, suffix: str) -> pd.DataFrame:
    """Load data from Google Cloud Storage bucket.

    Args:
        bucket_name (str): 
        suffix (str): Path inside the bucket

    Returns:
        pd.DataFrame: Df from parquet file.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(suffix)
    with blob.open("rb") as f:
        df = pd.read_parquet(f)
    return df


def upload_gcs(file_path: str, bucket_name: str, suffix: str) -> None:
    """Upload data to GCS.

    Args:
        filepath (str): File path to export.
        bucket_name (str): Bucket name.
        suffix (str): Path inside the bucket.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(suffix)
    blob.upload_from_filename(filename=file_path)

if __name__ == "__main__":
    main()
