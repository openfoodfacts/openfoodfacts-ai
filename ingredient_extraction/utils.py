import gzip
import html
import logging
from functools import cache
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import orjson
from tokenizers.pre_tokenizers import Punctuation, Sequence, WhitespaceSplit

from db import Annotation

logger = logging.getLogger(__name__)

ErrorType = tuple[str, str, str]


ID2LABEL = {
    0: "O",
    1: "B-ING",
    2: "I-ING",
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LABEL_LIST = list(ID2LABEL.values())


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
    mark_token: str = "b",
    html_escape: bool = False,
) -> str:
    highlighted_text = []
    previous_idx = 0
    escape_func = (lambda x: x) if html_escape is False else html.escape
    for start_idx, end_idx in offsets:
        highlighted_text.append(
            escape_func(text[previous_idx:start_idx])
            + f"<{mark_token}>"
            + escape_func(text[start_idx:end_idx])
            + f"</{mark_token}>"
        )
        previous_idx = end_idx
    highlighted_text.append(escape_func(text[previous_idx:]))
    return "".join(highlighted_text)


TokenizedOutputType = list[tuple[str, tuple[int, int]]]


def find_span_offsets(
    tokenized_output: TokenizedOutputType, start_idx: int, end_idx: int
) -> Optional[tuple[int, int]]:
    if not start_idx < end_idx:
        raise ValueError
    span_start_idx = None
    for token_idx, (_, (token_char_start_idx, token_char_end_idx)) in enumerate(
        tokenized_output
    ):
        if token_char_start_idx == start_idx:
            span_start_idx = token_idx
        if token_char_end_idx == end_idx:
            if span_start_idx is None:
                return None
            return span_start_idx, token_idx + 1
    return None


@cache
def get_pre_tokenizer():
    # In addition to XLM-Roberta tokenizer, we add punctuation split
    return Sequence([WhitespaceSplit(), Punctuation()])


def tokenize(text: str, offsets: list[tuple[int, int]]):
    # 27 errors
    logger.debug("offsets: %s", offsets)
    pre_tokenizer = get_pre_tokenizer()
    tokenized_output: TokenizedOutputType = pre_tokenizer.pre_tokenize_str(text)
    tokens = [t for t, _ in tokenized_output]
    spans = [
        find_span_offsets(tokenized_output, start_idx, end_idx)
        for start_idx, end_idx in sorted(set(offsets), key=lambda x: x[0])
    ]
    has_error = any(x is None for x in spans)

    if has_error:
        logger.debug("Tokenization error")
        logger.debug("text: '%s', offsets: %s", text, offsets)
        for span_idx, span in enumerate(spans):
            if span is None:
                char_start_idx, char_end_idx = offsets[span_idx]
                logger.debug(
                    "%d:%d, '%s'",
                    char_start_idx,
                    char_end_idx,
                    text[char_start_idx:char_end_idx],
                )
        logger.debug("----------")
        return None, None

    spans = [x for x in spans if x is not None]
    if not spans:
        ner_tags = [0 for _ in range(len(tokenized_output))]
    else:
        span_idx = 0
        span = spans[span_idx]
        ner_tags = []
        next_span = False

        logger.debug(f"current span: {span[0]}:{span[1]}")
        for i, (token, (token_char_start_idx, token_char_end_idx)) in enumerate(
            tokenized_output
        ):
            logger.debug(
                f"token {i}: '{token}' (char index: {token_char_start_idx}:{token_char_end_idx})"
            )
            if span is not None:
                span_start_idx, span_end_idx = span
                if span_start_idx >= i:
                    tag_id = 1 if span_start_idx == i else 0
                    logger.debug(f"TAG: '{ID2LABEL[tag_id]}'")
                    ner_tags.append(tag_id)
                    if span_end_idx - 1 == i:
                        next_span = True

                elif span_end_idx - 1 >= i:
                    tag_id = 2
                    logger.debug(f"TAG: '{ID2LABEL[tag_id]}'")
                    ner_tags.append(tag_id)
                    if span_end_idx - 1 == i:
                        next_span = True

                if next_span:
                    span_idx += 1
                    span = spans[span_idx] if span_idx < len(spans) else None
                    next_span = False
                    logger.debug(
                        f"current span: {span_start_idx}:{span_end_idx}"
                        if span
                        else None
                    )

            else:
                ner_tags.append(0)

        if span is not None:
            raise ValueError

    if len(tokens) != len(ner_tags):
        raise ValueError(
            "tokens (len=%d) and ner tags (length=%d) have different length:\n%s\n%s"
            % (len(tokens), len(ner_tags), tokens, ner_tags),
        )
    return tokens, ner_tags


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
