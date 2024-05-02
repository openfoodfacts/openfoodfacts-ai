import difflib
import os
from pathlib import Path
import logging
from typing import Mapping, Iterable
import json

from config.data import ArgillaConfig


def get_repo_dir():
    """Return the pathlib.Path of the Spellcheck repository"""
    return Path(os.path.dirname(__file__)).parent


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


def show_diff(original_text: str, corrected_text: str, deleted_element: str = ArgillaConfig.deleted_element):
    """Unify operations between two compared strings
    seqm is a difflib.SequenceMatcher instance whose a & b are strings
    """
    # Check if the process was not done
    if "<mark>" not in corrected_text:
        seqm = difflib.SequenceMatcher(None, original_text, corrected_text) 
        output= []
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.append(seqm.a[a0:a1])
            elif opcode == 'insert':
                output.append("<mark>" + seqm.b[b0:b1] + "</mark>")
            elif opcode == 'delete':
                output.append("<mark>" + deleted_element + "</mark>")
            elif opcode == 'replace':
                output.append("<mark>" + seqm.b[b0:b1] + "</mark>")
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
