"""VOC 2012 dataset class."""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datapoints

from data.dataset_classes import VOC_DICT

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}

class VOCDataset(Dataset):
    """VOC 2012 dataset class."""

    def __init__(
        self, root: str, image_set: str, transforms: Optional[Callable]
    ) -> None:
        super().__init__()

        voc_root = os.path.join(root, "VOCdevkit", "VOC")
        # voc_root1 = os.path.join(root, "VOCdevkit", "VOC_shot3")
        
        splits_dir = os.path.join(voc_root, "ImageSets", "Main")
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(os.path.join(root, "VOCdevkit", "VOC"), "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, "Annotations")
        self.targets = [os.path.join(target_dir, x + ".xml") for x in file_names]

        

        assert len(self.images) == len(self.targets)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def load_image(self, image_path: str) -> np.ndarray:
        """Load the image and apply the transformation."""
        # print('*** image_path = {}'.format(image_path))
        if not os.path.exists(image_path):
            raise ValueError("The image path {} does not exist.")

        # OpenCV load the image in the BGR format.
        # Rescale the pixel value from (0, 255) to (0, 1).
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # print("************* ", image.shape)

        return image

    def parse_voc_xml(
        self, xml_path: str
    ) -> Tuple[List[datapoints.BoundingBox], List[str]]:
        """Read the VOC metadata."""
        # print('************* ', xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_height = float(root.find("size/height").text)
        image_width = float(root.find("size/width").text)

        bounding_boxes = []
        categories = []
        for boxes in root.iter("object"):
            ymin = float(boxes.find("bndbox/ymin").text)
            xmin = float(boxes.find("bndbox/xmin").text)
            ymax = float(boxes.find("bndbox/ymax").text)
            xmax = float(boxes.find("bndbox/xmax").text)

            bounding_boxes.append(
                    datapoints.BoundingBox(
                        [xmin, ymin, xmax, ymax],
                        format=datapoints.BoundingBoxFormat.XYXY,
                        spatial_size=(image_height, image_width),
                    )
                )

            categories.append(VOC_DICT[boxes.find("name").text])
        return bounding_boxes, categories

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image = self.load_image(self.images[index])
        # print('******** ', self.images[index + 1])
        bounding_boxes, categories = self.parse_voc_xml(self.targets[index])
        # print('******** ', self.targets[index+ 1])

        image, bounding_boxes, categories = self.transforms(
            image, bounding_boxes, categories
        )

        sample = {"image": image, "boxes": bounding_boxes, "labels": categories}

        return sample
