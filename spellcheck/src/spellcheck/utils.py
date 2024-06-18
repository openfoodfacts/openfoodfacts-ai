import difflib
import os
from pathlib import Path
import logging
from typing import Mapping, Iterable, Literal
import json
import time

from spellcheck.config import ArgillaConfig


def get_repo_dir():
    """Return the pathlib.Path of the Spellcheck repository"""
    return Path(os.path.dirname(__file__)).parent.parent


def get_logger(level: str = "INFO") -> logging.Logger:
    """Return LOGGER

    Args:
        level (str, optional): Logging level. Defaults to "INFO".

    Returns:
        logging.Logger: Logger
    """
    logging.basicConfig(
        level=logging.getLevelName(level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def show_diff(
    original_text: str, 
    corrected_text: str,
    color: Literal["yellow", "orange", "red"] = "yellow",
    deleted_element: str = ArgillaConfig.deleted_element
) -> str:
    """Unify operations between two compared strings
    seqm is a difflib.SequenceMatcher instance whose a & b are strings

    Args:
        original_text (str): Comparison reference
        corrected_text (str): Text to highlight differences
        color (Literal["yellow", "orange", "red"], optional): Highlight color. Defaults to "yellow".
        deleted_element (str, optional): _description_. Defaults to ArgillaConfig.deleted_element.

    Raises:
        RuntimeError: unexpected opcode

    Returns:
        str: highlighted text
    """
    html_color = ArgillaConfig.html_colors[color]
    # Check if the process was not done
    if "<mark>" not in corrected_text:
        seqm = difflib.SequenceMatcher(None, original_text, corrected_text) 
        output= []
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.append(seqm.a[a0:a1])
            elif opcode == 'insert':
                output.append(f"<mark style=background-color:{html_color}>" + seqm.b[b0:b1] + "</mark>")
            elif opcode == 'delete':
                output.append(f"<mark style=background-color:{html_color}>" + deleted_element + "</mark>")
            elif opcode == 'replace':
                output.append(f"<mark style=background-color:{html_color}>" + seqm.b[b0:b1] + "</mark>")
            else:
                raise RuntimeError("unexpected opcode")
        return ''.join(output)
    else:
        return corrected_text


def load_jsonl(path: Path) -> Iterable[Mapping]:
    """Load JSONL file
    Args:
        path (Path): JSONL path
    Raise:
        ValueError: Not a jsonl file.
    Returns:
        Iterable[Mapping]: Data
    """
    if path.suffix != ".jsonl":
        raise ValueError(f"'.jsonl' file expected. Got {path.suffix} instead.")
    with open(path, "r") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def timer(fn):
    """Decorator to track function duration."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        timestamp = time.time()
        logger.info(f"Start {fn.__name__}.")
        output = fn(*args, **kwargs)
        logger.info(f"The function {fn.__name__} took {round(time.time() - timestamp)} to finish.")
        return output
    return wrapper