import albumentations as A
import cv2
import torch
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from ultralytics.models.yolo.classify import ClassificationPredictor

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


def get_train_transform(
    max_size: int,
    square_symmetry_prob: float = 1.0,
    coarse_dropout_prob: float = 0.4,
    drop_color_prob: float = 0.1,
):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size, p=1.0),
            A.PadIfNeeded(min_height=max_size, min_width=max_size, p=1.0),
            A.SquareSymmetry(p=square_symmetry_prob),
            A.CoarseDropout(p=coarse_dropout_prob),
            A.OneOf(
                [
                    A.ToGray(p=1.0),  # p=1.0 inside OneOf
                    A.ChannelDropout(p=1.0),  # p=1.0 inside OneOf
                ],
                p=drop_color_prob,
            ),
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


class CustomClassificationPredictor(ClassificationPredictor):
    def setup_source(self, source):
        super().setup_source(source)
        self.transforms = get_predict_transform(self.args.imgsz)

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [
                    self.transforms(
                        image=Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    )["image"]
                    for im in img
                ],
                dim=0,
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(
            self.model.device
        )
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
