"""Training and validation dataset initialization."""
import argparse
from typing import Any, Dict, List, Tuple, Union

import torch
import torchvision
from torchvision import datapoints


from data.transforms import build_transforms
from data.voc_dataset import VOCDataset

DATASET_NAMES = ["voc"]


def build_dataset(
    configs: Dict[str, Any],
    dataset_type: str,
) -> torch.utils.data.Dataset:
    """Build the dataset based on the input specs."""
    dataset_name = configs["dataset"]["name"]

    if dataset_name not in DATASET_NAMES:
        raise ValueError("Unsupported dataset: {}.".format(dataset_name))
    if dataset_type not in ("train", "val", "test"):
        raise ValueError("Unsupported dataset type: {}.".format(dataset_type))

    if dataset_name == "voc":
        dataset = VOCDataset(
            root=configs["dataset"]["path"],
            image_set=dataset_type,
            transforms=build_transforms(
                dataset_type, configs[dataset_type]["crop_size"]
            ),
        )
        
    return dataset


def collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, List[Dict[str, Union[torch.Tensor, torchvision.datapoints.BoundingBox]]]]:
    """Collation function for dataloader as the length of labels may be different."""
    images = []
    targets = []

    for b in batch:
        images.append(b["image"])
        targets.append(
            {
                "boxes": torch.stack([dp.data for dp in b["boxes"]], dim=0),
                "labels": torch.Tensor(b["labels"]).long(),
            }
        )

    images = torch.stack(images, dim=0)

    return images, targets
