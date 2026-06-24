import re
from difflib import Differ
from pathlib import Path

import orjson
from deepdiff import DeepHash
from pydantic import BaseModel
from pydantic_ai.capabilities.thinking import ThinkingLevel


class ModelOutputCache[BaseModelType: BaseModel | str]:
    def __init__(
        self,
        task_name: str,
        output_type: type[BaseModelType],
        cache_dir: Path | None = None,
    ):
        if cache_dir is None:
            cache_dir = Path("~/.cache/llm_evals").expanduser()
        self.cache_dir = cache_dir
        self.task_name = task_name
        self.output_type = output_type

    def get_query_cache_path(
        self,
        *,
        model: str,
        instructions: str | list,
        output_mode: str | None,
        thinking_effort: ThinkingLevel,
    ) -> Path:
        if self.output_type is str:
            json_schema = None
        else:
            json_schema = self.output_type.model_json_schema()
        model = model.replace("/", "_")
        cache_key = (
            model,
            self.task_name,
            instructions,
            json_schema,
            output_mode,
            thinking_effort,
        )
        cache_sha256 = DeepHash(cache_key)[cache_key]

        # Split the cache sha256 into subdirectories for better file system
        # performance
        cache_sha256_str = str(cache_sha256)
        subdirs = [cache_sha256_str[i : i + 2] for i in range(0, 6, 2)]
        cache_subdir = Path(*subdirs)
        full_cache_dir = self.cache_dir / self.task_name / model / cache_subdir
        return full_cache_dir / f"{cache_sha256}.json"

    def check_cache(
        self,
        *,
        model: str,
        instructions: str | list,
        output_mode: str | None,
        thinking_effort: ThinkingLevel,
    ) -> BaseModelType | None:
        query_cache_path = self.get_query_cache_path(
            model=model,
            instructions=instructions,
            output_mode=output_mode,
            thinking_effort=thinking_effort,
        )
        if query_cache_path.exists():
            output = orjson.loads(query_cache_path.read_bytes())["output"]
            if self.output_type is str:
                return output
            else:
                return self.output_type.model_validate(output)
        return None

    def save_to_cache(
        self,
        *,
        model: str,
        instructions: str | list,
        output_mode: str | None,
        thinking_effort: ThinkingLevel,
        output: BaseModelType,
    ) -> None:
        query_cache_path = self.get_query_cache_path(
            model=model,
            instructions=instructions,
            output_mode=output_mode,
            thinking_effort=thinking_effort,
        )
        query_cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_type is str:
            _output = output
            json_schema = None
        else:
            _output = output.model_dump()
            json_schema = self.output_type.model_json_schema()
        data = {
            "output": _output,
            "model": model,
            "task_name": self.task_name,
            "thinking_effort": thinking_effort,
            "instructions": instructions,
            "output_mode": output_mode,
            "json_schema": json_schema,
        }
        with query_cache_path.open("wb") as f:
            f.write(orjson.dumps(data))


def normalize(s: str) -> str:
    return (
        # Replace all whitespace characters (including tabs, newlines, etc.) with a single space
        re.sub(r"\s+", " ", s)
        # Normalize quotes
        .replace("’", "'")
        .replace("œ", "oe")
        # Remove leading/trailing whitespace
        .strip()
    )


def tokenize(s: str, split_chars: str = ".:,;!/()[]{}") -> list[str]:
    """Tokenize the input string.

    The input string is first splitted into tokens using whitespace as separator,
    then we look for any chars in `split_chars` at the beginning or the end of
    the token to further split into additional tokens.
    """
    output = []
    for token in s.split(" "):
        if len(token) > 1:
            while len(token) > 1 and any(
                token.startswith(char) for char in split_chars
            ):
                output.append(token[0])
                token = token[1:]

            token_buffer = []
            while len(token) > 1 and any(token.endswith(char) for char in split_chars):
                token_buffer.insert(0, token[-1])
                token = token[:-1]
            output.append(token)
            if token_buffer:
                output += token_buffer
        else:
            output.append(token)

    return output


def get_diff(s1: str, s2: str) -> str:
    """Compute the diff between two strings, returning the diff as a multi-line string.

    Words are first splitted using whitespace as separator, then they are compared using
    Differ.
    """
    expected_lines = tokenize(s1)
    actual_lines = tokenize(s2)
    diffs = list(Differ().compare(expected_lines, actual_lines))
    # Filter to only lines that differ
    return "\n".join([d.replace("\n", "") for d in diffs if not d.startswith("  ")])
