import argparse
from typing import List

from vllm import LLM, SamplingParams


def parser() -> argparse.Namespace:
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Spellcheck module.")
    parser.add_argument("file_path", type=str, default="path_to_the_file", help="Path to the file.")
    parser.add_argument("model_path", default="model/", type=str, help="Model path.")
    parser.add_argument("max_model_len", default=1024, type=int, help="Maximum model context length. A lower max context length reduces the memory footprint and accelerate the inference.")
    parser.add_argument("temperature", default=0, type=float, help="Sampling temperature.")
    parser.add_argument("max_tokens", default=1024, type=int, help="Maximum number of tokens to generate.")
    parser.add_argument("quantization", default="fp8", type=str, help="Quantization type.")
    parser.add_argument("dtype", default="auto", type=str, help="Model weights precision. Default corresponds to the modle config (float16 here)")
    return parser.parse_args()


def main() -> None:
    """Batch processing.
    """
    args = parser()
    texts = load_inputs(args.file_path)
    instructions = [prepare_instruction(text) for text in texts]    
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
    corrections = batch_inference(instructions, llm=llm, sampling_params=sampling_params)
    upload_outputs(args.output_file_paths, corrections)  



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


def load_inputs(file_path: str) -> List[str]:
    """Load data from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        List[str]: List of ingredients.
    """
    return ["patata 15%, piment 15,2%, pamplemousse 47 %, aubergine 18.1"]


def upload_outputs(output_path, outputs):
    """_summary_

    Args:
        output_path (_type_): _description_
    """
    pass



def batch_inference(
        texts: List[str], 
        llm: LLM,
        sampling_params: SamplingParams
    ) -> List[str]:
    """_summary_

    Args:
        texts (List[str]): _description_

    Returns:
        List[str]: _description_
    """
    outputs = llm.generate(texts, sampling_params,)
    corrections = [output.outputs[0].text for output in outputs]
    return corrections


if __name__ == "__main__":
    parser()