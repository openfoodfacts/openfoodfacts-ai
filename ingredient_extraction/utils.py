import gzip
import html
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Optional, Union
import logging
from urllib.parse import urlparse

from uniseg import wordbreak
import orjson
from spacy.tokens import Doc
from db import Annotation

logger = logging.getLogger(__name__)

ErrorType = tuple[str, str, str]


def find_span_offsets(text: str, substring: str):
    start_idx = text.find(substring)
    if start_idx == -1:
        raise ValueError()
    return start_idx, start_idx + len(substring)


def get_root_logger(logging_level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging_level)
    logger.addHandler(handler)
    return logger


def generate_identifier(barcode: str, image_id: str, prompt_version: str):
    return f"{barcode}_{image_id}_prompt_version-{prompt_version}"


def fetch_annotations() -> dict[str, dict]:
    return {
        annotation["identifier"]: annotation
        for annotation in Annotation.select().dicts()
    }


def generate_highlighted_text(
    text: str,
    offsets: list[tuple[int, int]],
    start_token: str = "<b>",
    end_token: str = "</b>",
    html_escape: bool = False,
) -> str:
    highlighted_text = []
    previous_idx = 0
    escape_func = (lambda x: x) if html_escape is False else html.escape
    for start_idx, end_idx in offsets:
        highlighted_text.append(
            escape_func(text[previous_idx:start_idx])
            + start_token
            + escape_func(text[start_idx:end_idx])
            + end_token
        )
        previous_idx = end_idx
    highlighted_text.append(escape_func(text[previous_idx:]))
    return "".join(highlighted_text)


def find_span_offsets(
    tokens: list[str], start_idx: int, end_idx: int
) -> Optional[tuple[int, int]]:
    offset = 0
    if not start_idx < end_idx:
        breakpoint()
        raise ValueError
    span_start_idx = None
    for token_idx, token in enumerate(tokens):
        if offset == start_idx:
            span_start_idx = token_idx
        offset += len(token)
        if offset == end_idx:
            if span_start_idx is None:
                return None
            return span_start_idx, token_idx + 1
    return None


def tailor(string: str, breakables: Iterable[int]) -> Iterable[int]:
    new_breakables = []
    next_break = False
    for s, b in zip(string, breakables):
        if s == ":":
            b = 1
            next_break = True
        else:
            if next_break:
                b = 1
            next_break = False
        new_breakables.append(b)
    return new_breakables


def tokenize(text: str, offsets: list[tuple[int, int]]):
    logger.debug("offsets: %s", offsets)
    tokens: list[str] = list(wordbreak.words(text, tailor=tailor))
    non_space_tokens = [t for t in tokens if not t.isspace()]
    spans = [
        find_span_offsets(tokens, start_idx, end_idx)
        for start_idx, end_idx in sorted(set(offsets), key=lambda x: x[0])
    ]
    has_error = any(x is None for x in spans)

    if has_error:
        logger.debug("text: '%s', offsets: %s", text, offsets)
        return None, None

    spans = [x for x in spans if x is not None]
    if not spans:
        ner_tags = [0 for _ in range(len(non_space_tokens))]
    else:
        span_idx = 0
        span = spans[span_idx]
        ner_tags = []
        logger.debug(f"current span: {span[0]}:{span[1]}")
        char_offset = 0
        for i, token in enumerate(tokens):
            logger.debug(f"token {i}: '{token}' (char index: {char_offset})")
            if span is not None:
                span_start_idx, span_end_idx = span
                if span_start_idx > i:
                    logger.debug("TAG: 'O'")
                    ner_tags.append(0)
                elif span_start_idx == i:
                    logger.debug("TAG: 'B-ING'")
                    ner_tags.append(1)
                elif span_end_idx > i:
                    logger.debug("TAG: 'I-ING'")
                    ner_tags.append(2)
                    if i == len(tokens) - 1:
                        # Last token in document
                        span_idx += 1
                        span = spans[span_idx] if span_idx < len(spans) else None
                        if span:
                            logger.debug(
                                f"current span: {span_start_idx}:{span_end_idx}"
                            )
                        else:
                            logger.debug("No span left")
                elif span_end_idx == i:
                    logger.debug("TAG: 'O'")
                    ner_tags.append(0)
                    span_idx += 1
                    span = spans[span_idx] if span_idx < len(spans) else None
                    if span:
                        logger.debug(f"current span: {span_start_idx}:{span_end_idx}")
                    else:
                        logger.debug("No span left")
                else:
                    raise ValueError
            else:
                ner_tags.append(0)
            char_offset += len(token)

        if span is not None:
            raise ValueError

    return non_space_tokens, ner_tags


def get_open_fn(filepath: Union[str, Path]) -> Callable:
    filepath = str(filepath)
    if filepath.endswith(".gz"):
        return gzip.open
    else:
        return open


def jsonl_iter(jsonl_path: Union[str, Path]) -> Iterable[dict]:
    """Iterate over elements of a JSONL file.

    :param jsonl_path: the path of the JSONL file. Both plain (.jsonl) and
        gzipped (jsonl.gz) files are supported.
    :yield: dict contained in the JSONL file
    """
    open_fn = get_open_fn(jsonl_path)

    with open_fn(str(jsonl_path), "rt", encoding="utf-8") as f:
        yield from jsonl_iter_fp(f)


def gzip_jsonl_iter(jsonl_path: Union[str, Path]) -> Iterable[dict]:
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
        yield from jsonl_iter_fp(f)


def jsonl_iter_fp(fp) -> Iterable[dict]:
    for line in fp:
        line = line.strip("\n")
        if line:
            yield orjson.loads(line)
