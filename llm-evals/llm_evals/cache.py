import functools
import json
from pathlib import Path

from deepdiff import DeepHash

CACHE_DIR = Path("~/.cache/llm_evals").expanduser()


def get_query_cache_path(model: str, task_name: str, args: tuple, kwargs: dict) -> Path:
    cache_key = (task_name, args, kwargs)
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
        model = kwargs["model"]
        task_name = kwargs["task_name"]
        # Implement caching logic here
        query_cache_path = get_query_cache_path(model, task_name, args, kwargs)

        # Check if result is in cache
        if (resp := _load_cached_response(query_cache_path)) is not None:
            return resp

        # If not, call the function and store the result in cache
        result = await func(*args, **kwargs)
        data = {"output": result, "model": model, "task_name": task_name}
        _save_cached_response(query_cache_path, data)
        return result

    return wrapper


def cache_llm_request_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = kwargs["model"]
        task_name = kwargs["task_name"]
        # Implement caching logic here
        query_cache_path = get_query_cache_path(model, task_name, args, kwargs)

        # Check if result is in cache
        if (resp := _load_cached_response(query_cache_path)) is not None:
            print(resp)
            return resp

        # If not, call the function and store the result in cache
        result = func(*args, **kwargs)
        data = {"output": result, "model": model, "task_name": task_name}
        _save_cached_response(query_cache_path, data)
        return result

    return wrapper
