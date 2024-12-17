"""Data transformations."""
from typing import Any, List, Tuple

import torch
from torchvision import datapoints
import torchvision.transforms.v2 as transforms


def build_transforms(
    dataset_type: str = "train", crop_size: int = 256
) -> transforms.Compose:
    """Build data augmentation for training/evaluation."""
    tfs = []
    tfs.append(transforms.ToImageTensor())
    tfs.append(transforms.ConvertImageDtype(torch.float32))
    if dataset_type == "train":
        tfs.append(transforms.ColorJitter(contrast=0.5))
        tfs.append(transforms.RandomHorizontalFlip(p=0.5))
        tfs.append(
            transforms.RandomResizedCrop(size=(crop_size, crop_size), antialias=True)
        )
    else:
        if crop_size > 0:
            tfs.append(transforms.CenterCrop(size=(crop_size, crop_size)))
    tfs.append(RemoveInvalidBoxes())

    return transforms.Compose(tfs)


class RemoveInvalidBoxes(torch.nn.Module):
    """Remove invalid bounding boxes after cropping."""

    def __init__(self, min_size: int = 1) -> None:
        super().__init__()

        self.min_size = min_size

    def forward(
        self, inputs: Any
    ) -> Tuple[torch.Tensor, List[datapoints.BoundingBox], List[int]]:
        img: torch.Tensor = inputs[0]
        bboxes: List[datapoints.BoundingBox] = inputs[1]
        labels: List[int] = inputs[2]

        sanitized_bboxes = []
        sanitized_labels = []

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox.data
            bbox_size = (x2 - x1) * (y2 - y1)

            if bbox_size > self.min_size:
                sanitized_bboxes.append(bbox)
                sanitized_labels.append(label)

        if len(sanitized_bboxes) == 0:
            sanitized_bboxes.append(
                datapoints.BoundingBox(
                    [0, 0, img.shape[1], img.shape[2]],
                    format=datapoints.BoundingBoxFormat.XYXY,
                    spatial_size=(img.shape[1], img.shape[2]),
                )
            )
            sanitized_labels.append(-1)

        return img, sanitized_bboxes, sanitized_labels
