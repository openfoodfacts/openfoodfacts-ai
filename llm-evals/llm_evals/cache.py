import functools
import json
from pathlib import Path

from deepdiff import DeepHash

CACHE_DIR = Path("~/.cache/llm_evals").expanduser()


def get_query_cache_path(
    *,
    image_url: str,
    model: str,
    task_name: str,
    instructions: str,
    json_schema: str,
    output_mode: str,
) -> Path:
    model = model.replace("/", "_")
    cache_key = (image_url, model, task_name, instructions, json_schema, output_mode)
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
        image_url = kwargs.get("image_url")
        image_urls = kwargs.get("image_urls")

        if image_urls is not None:
            if image_url is not None:
                raise ValueError("Provide either image_url or image_urls, not both.")
            image_urls_str = ",".join(image_urls)
        else:
            if image_url is None:
                raise ValueError("Either image_url or image_urls must be provided.")
            image_urls_str = image_url

        model = kwargs["model"]
        task_name = kwargs["task_name"]
        instructions = kwargs["instructions"]
        json_schema = kwargs["json_schema"]
        output_mode = kwargs["output_mode"]
        # Implement caching logic here
        query_cache_path = get_query_cache_path(
            image_url=image_urls_str,
            model=model,
            task_name=task_name,
            instructions=instructions,
            json_schema=json_schema,
            output_mode=output_mode,
        )

        # Check if result is in cache
        if (resp := _load_cached_response(query_cache_path)) is not None:
            return resp

        # If not, call the function and store the result in cache
        result = await func(*args, **kwargs)
        data = {
            "image_url": image_url,
            "output": result,
            "model": model,
            "task_name": task_name,
            "instructions": instructions,
            "output_mode": output_mode,
            "json_schema": json.loads(json_schema),
        }
        _save_cached_response(query_cache_path, data)
        return result

    return wrapper
