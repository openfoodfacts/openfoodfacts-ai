import pathlib
import random
from typing import List

from annoy import AnnoyIndex
from keras.applications import resnet50
import numpy as np
from PIL import Image


BASE_DIR = pathlib.Path(__name__).parent.resolve()
MODEL_TYPE = "ResNet50"
EMBEDDING_SIZE = 2048
TREE_COUNT = 100
INDEX_OUTPUT_PATH = BASE_DIR / "index.annoy"
KEYS_OUTPUT_PATH = BASE_DIR / "index.keys"


def load_index(index_path: pathlib.Path) -> AnnoyIndex:
    index = AnnoyIndex(EMBEDDING_SIZE, "euclidean")
    print("Loading index from {}".format(index_path))
    index.load(str(index_path), prefault=True)
    return index


def load_keys(key_path: pathlib.Path) -> List[str]:
    keys = []
    with key_path.open("r") as f:
        for key in f:
            key = key.strip("\n")
            keys.append(key)

    return keys


def find_nearest_neighbor(
    index: AnnoyIndex,
    model,
    preprocess_fn,
    image: Image.Image,
    keys: List[str],
    count: int,
):
    ref_embedding = generate_embedding(model, image, preprocess_fn)
    ids, distances = index.get_nns_by_vector(
        ref_embedding, n=count, include_distances=True
    )
    external_ids = [keys[id_] for id_ in ids]
    return list(zip(external_ids, distances))


def build_model(model_type: str):
    if model_type == "ResNet50":
        return resnet50.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling="max",
        )

    raise ValueError("unknown model type: {}".format(model_type))


def get_preprocess_fn(model_type: str):
    if model_type == "ResNet50":
        return resnet50.preprocess_input

    raise ValueError("unknown model type: {}".format(model_type))


def generate_embedding(model, image, preprocess_fn) -> np.ndarray:
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, 0)
    image_array = preprocess_fn(image_array)
    return model.predict(image_array)[0]


def get_random_image(image_dir: pathlib.Path):
    image_path: pathlib.Path = random.choice(list(image_dir.glob("*.jpg")))
    return image_path, Image.open(image_path)


def generate_embedding_from_dir(model, image_dir: pathlib.Path, preprocess_fn):
    for image_path in image_dir.glob("*.jpg"):
        image = Image.open(image_path)
        embedding = generate_embedding(model, image, preprocess_fn)
        yield image_path, embedding


def build_index(
    model,
    preprocess_fn,
    image_dir: pathlib.Path,
    index_output_path: pathlib.Path,
    keys_output_path: pathlib.Path,
    embedding_size: int,
    tree_count: int,
):
    index = AnnoyIndex(embedding_size, "euclidean")

    offset: int = 0
    image_names = []

    for image_path, embedding in generate_embedding_from_dir(
        model, image_dir, preprocess_fn
    ):
        index.add_item(offset, embedding)
        offset += 1
        image_names.append(image_path.name)

    index.build(tree_count)
    index.save(str(index_output_path))

    with keys_output_path.open("w") as f:
        for name in image_names:
            f.write("{}\n".format(name))


def generate_examples():
    index = load_index(INDEX_OUTPUT_PATH)
    keys = load_keys(KEYS_OUTPUT_PATH)

    for i in range(20):
        random_image_path, random_image = get_random_image(IMAGE_DIR)
        example_dir = EXAMPLE_DIR / random_image_path.stem

        if example_dir.is_dir():
            continue

        example_dir.mkdir(exist_ok=True, parents=True)

        for neighbor_name, distance in find_nearest_neighbor(
            index=index,
            model=model,
            preprocess_fn=preprocess_fn,
            image=random_image,
            keys=keys,
            count=30,
        ):
            output_path = example_dir / "{}-{}.jpg".format(distance, neighbor_name)
            output_path.symlink_to(IMAGE_DIR / "{}.jpg".format(neighbor_name))


if __name__ == "__main__":
    model = build_model(MODEL_TYPE)
    preprocess_fn = get_preprocess_fn(MODEL_TYPE)
    IMAGE_DIR = BASE_DIR / "bounding_box"
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    INDEX_OUTPUT_PATH = BASE_DIR / "index.annoy"
    KEYS_OUTPUT_PATH = BASE_DIR / "index.keys"
    EXAMPLE_DIR = BASE_DIR / "examples"
    # build_index(
    #    model,
    #    preprocess_fn,
    #    IMAGE_DIR,
    #    INDEX_OUTPUT_PATH,
    #    KEYS_OUTPUT_PATH,
    #    EMBEDDING_SIZE,
    #    TREE_COUNT,
    # )
    generate_examples()
