import functools
import json
from pathlib import Path

from deepdiff import DeepHash

CACHE_DIR = Path("~/.cache/llm_evals").expanduser()


def get_query_cache_path(
    *,
    image_urls: list[str],
    model: str,
    task_name: str,
    instructions: str,
    json_schema: str,
    output_mode: str,
    thinking_config: str | None,
) -> Path:
    model = model.replace("/", "_")
    image_urls_str = ",".join(image_urls)
    cache_key = (
        image_urls_str,
        model,
        task_name,
        instructions,
        json_schema,
        output_mode,
        thinking_config,
    )
    cache_sha256 = DeepHash(cache_key)[cache_key]

    # Split the cache sha256 into subdirectories for better file system
    # performance
    cache_sha256_str = str(cache_sha256)
    subdirs = [cache_sha256_str[i : i + 2] for i in range(0, 6, 2)]
    cache_subdir = Path(*subdirs)
    full_cache_dir = CACHE_DIR / task_name / model / cache_subdir
    return full_cache_dir / f"{cache_sha256}.json"


def _load_cached_response(query_cache_path: Path) -> str | None:
    if query_cache_path.exists():
        with query_cache_path.open("r") as f:
            data = json.loads(f.read())
            return data["output"]
    return None


def _save_cached_response(query_cache_path: Path, data: dict) -> None:
    query_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with query_cache_path.open("w") as f:
        json.dump(data, f)


def cache_llm_request_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        image_urls = kwargs["image_urls"]
        model = kwargs["model"]
        task_name = kwargs["task_name"]
        instructions = kwargs["instructions"]
        json_schema = kwargs["json_schema"]
        output_mode = kwargs["output_mode"]
        thinking_config = kwargs["thinking_config"]
        # Implement caching logic here
        query_cache_path = get_query_cache_path(
            image_urls=image_urls,
            model=model,
            task_name=task_name,
            instructions=instructions,
            json_schema=json_schema,
            output_mode=output_mode,
            thinking_config=thinking_config,
        )

        # Check if result is in cache
        if (resp := _load_cached_response(query_cache_path)) is not None:
            return resp

        # If not, call the function and store the result in cache
        result = await func(*args, **kwargs)
        data = {
            "image_urls": image_urls,
            "output": result,
            "model": model,
            "task_name": task_name,
            "thinking_config": thinking_config,
            "instructions": instructions,
            "output_mode": output_mode,
            "json_schema": json.loads(json_schema),
        }
        _save_cached_response(query_cache_path, data)
        return result

    return wrapper
