# /// script
# dependencies = [
#   "typer",
#   "tqdm",
#   "Pillow",
#   "ultralytics",
#   "albumentations",
# ]
# ///

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import typer
import ultralytics
from albumentations.pytorch.transforms import ToTensorV2
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


def get_train_transform(
    max_size: int, square_symmetry_prob: float = 1.0, coarse_dropout_prob: float = 0.4
):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size, p=1.0),
            A.PadIfNeeded(min_height=max_size, min_width=max_size, p=1.0),
            A.SquareSymmetry(p=square_symmetry_prob),
            A.CoarseDropout(p=coarse_dropout_prob),
            A.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


def get_predict_transform(max_size: int):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size, p=1.0),
            A.PadIfNeeded(min_height=max_size, min_width=max_size, p=1.0),
            A.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


class CustomizedDataset(ClassificationDataset):
    """A customized dataset class for image classification with enhanced data
    augmentation transforms."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """Initialize a customized classification dataset with enhanced data
        augmentation transforms."""
        super().__init__(root, args, augment, prefix)
        train_transforms = get_train_transform(args.imgsz)
        val_transforms = get_predict_transform(args.imgsz)
        self.torch_transforms = train_transforms if augment else val_transforms

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[
            i
        ]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if (
                im is None
            ):  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample = self.torch_transforms(image=im)["image"]
        return {"img": sample, "cls": j}


class CustomizedTrainer(ClassificationTrainer):
    """A customized trainer class for YOLO classification models with enhanced
    dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build a customized dataset for classification training or
        validation."""
        return CustomizedDataset(
            root=img_path, args=self.args, augment=mode == "train", prefix=mode
        )


def main(
    dataset_dir: Path,
    model_name: str = "yolo11n-cls.pt",
    epochs: int = 10,
    imgsz: int = 224,
    batch: int = 64,
):
    model = ultralytics.YOLO(model_name)
    model.train(
        data=dataset_dir,
        trainer=CustomizedTrainer,
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
    )


if __name__ == "__main__":
    typer.run(main)
