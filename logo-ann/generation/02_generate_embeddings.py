import argparse
from email.policy import default
import pathlib
from typing import Iterable, Set, Any

from efficientnet_pytorch import EfficientNet
from transformers import CLIPModel, CLIPProcessor
import h5py
from more_itertools import chunked
import numpy as np
import tqdm
import torch
import PIL

from utils import get_offset, get_seen_set

"""Return a hdf5 file containing the embeddings of every logo of the input hdf5 file.

> > >  python3 02_generate_ebeddings.py data_path output_path (--batch-size n) (--min-confidence m) --model-type str

data_path: path of the hdf5 file containing the data regarding the logos
output_path: path where the hdf5 file of the embeddings and the external ids will be returned
batch-size: size of each batche of logos embedded at the same time 
min-confidence: minimum of confidence allowed for a logo to be accepted as one
model-type: name of the specific model used
"""


def build_model(model_type: str):
    return CLIPModel.from_pretrained(f"openai/{model_type}").vision_model
    # return EfficientNet.from_pretrained(model_type)


def get_output_dim(model_type: str):

    """Return the embeddings size according to the model used."""

    if model_type == "efficientnet-b0":
        return 1280

    if model_type == "efficientnet-b5":
        return 2048

    if model_type == "clip-vit-base-patch16" or model_type == "clip-vit-base-patch32":
        return 768

    if model_type == "clip-vit-large-patch14":
        return 1024

    raise ValueError("unknown model type: {}".format(model_type))


def get_item_count(file_path: pathlib.Path) -> int:
    with h5py.File(str(file_path), "r") as f:
        image_dset = f["image"]
        return len(image_dset)


def generate_embeddings_iter(
    model,
    file_path: pathlib.Path,
    batch_size: int,
    device: torch.device,
    seen_set: Set[int],
    min_confidence: float = 0.5,
    processor: Any = None,
):

    """Inputs:

    - model: name of the specific model used
    - file_path: path of the hdf5 file containing the data of all the logos
    - batch-size: size of each batche of logos embedded at the same time 
    - device: hardware used to compute the embeddings
    - seen_set: set of every logo already embedded in 
    - min-confidence: minimum of confidence allowed for a logo to be accepted as one

    Yield the following outputs:
    - embeddings: embeddings of every logo of the yielded batch 
    - external_id: id of the logo
    """

    with h5py.File(str(file_path), "r") as f:
        image_dset = f["image"]
        confidence_dset = f["confidence"]
        external_id_dset = f["external_id"]

        for slicing in chunked(range(len(image_dset)), batch_size):
            slicing = np.array(
                slicing
            )  # slicing was a list, we make an array out of it
            external_ids = external_id_dset[slicing]
            mask = external_ids == 0

            if np.all(
                mask
            ):  # if we only have zeros as external ideas, it's an empty batch and it means we won't have any other logos so we stop the loop
                break

            mask = (~mask) & (
                confidence_dset[slicing] >= min_confidence
            )  # we add to the mask the confidence parameter to avoid embedding

            for i, external_id in enumerate(external_ids):
                if int(external_id) in seen_set:
                    mask[i] = 0

            if np.all(~mask):  # if we only have zeros at this step, we have a batch only with empty data or already seen logos

                continue

            images = image_dset[slicing][mask]
            images = np.moveaxis(images, -1, 1)  # move channel dim to 1st dim

            """### If using efficientnet models :
            with torch.no_grad():
                torch_images = torch.tensor(images, dtype=torch.float32, device=device)
                embeddings = model.extract_features(torch_images).cpu().numpy()
            

            max_embeddings = np.max(embeddings, (-1, -2))
            yield (
                max_embeddings,
                external_ids[mask],
            )
            ###
            """

            ### If using CLIP models :
            with torch.no_grad():
                # preprocess the images to put them into the model
                images = processor(
                    images=[PIL.Image.fromarray(images[i], mode="RGB") for i in range(batch_size)],  # convert the np.array to PIL in order to use the CLIProcessor
                    return_tensors="pt",
                )[
                    "pixel_values"
                ]

                embeddings = (
                    model(**{"pixel_values": images.to(device)})
                    .pooler_output.cpu()
                    .numpy()
                )  # generate the embeddings
                if np.any(np.isnan(embeddings)):  # checking that the values are not NaN
                    print("A NaN value was detected, avoiding the loop")
                    continue

            yield (embeddings, external_ids[mask])
            ###


def generate_embedding_from_hdf5(
    data_gen: Iterable, output_path: pathlib.Path, output_dim: int, count: int
):

    """Save the embedding and the external id of each logo (data in data_gen) in an hdf5 file (the output_path).

    - data_gen: yielded embeddings and external ids of each logo from generate_embeddings_iter
    - output_path: path of the output hdf5 file
    - output_dim: dimension of the embeddings (depends on the computer vision model used)
    - count: amount of embeddings you want to save 
    """

    file_exists = output_path.is_file()

    with h5py.File(str(output_path), "a") as f:
        if not file_exists:
            embedding_dset = f.create_dataset(
                "embedding", (count, output_dim), dtype="f", chunks=True
            )
            external_id_dset = f.create_dataset(
                "external_id", (count,), dtype="i", chunks=True
            )
            offset = 0
        else:
            offset = get_offset(f)
            embedding_dset = f["embedding"]
            external_id_dset = f["external_id"]

        print("Offset: {}".format(offset))

        for (embeddings_batch, external_id_batch) in data_gen:
            slicing = slice(offset, offset + len(embeddings_batch))
            embedding_dset[slicing] = embeddings_batch
            external_id_dset[slicing] = external_id_batch
            offset += len(embeddings_batch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--model-type", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.data_path.is_file()
    assert not args.output_path.is_file()
    assert (
        args.model_type.startswith("clip-vit-base-patch")
        and args.model_type[-1].isdigit()
    )

    model_type = args.model_type
    model = build_model(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained(f"openai/{model_type}")

    seen_set = get_seen_set(args.output_path)
    print("Number of seen items: {}".format(len(seen_set)))

    data_gen = tqdm.tqdm(
        generate_embeddings_iter(
            model,
            args.data_path,
            args.batch_size,
            device,
            seen_set,
            args.min_confidence,
            processor,
        )
    )
    output_dim = get_output_dim(model_type)
    count = get_item_count(args.data_path)
    generate_embedding_from_hdf5(data_gen, args.output_path, output_dim, count)
