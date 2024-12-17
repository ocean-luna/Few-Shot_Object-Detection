"""Utility functions for model training and validation."""
import argparse
from datetime import datetime
import os
from typing import Any, Dict, List, Tuple

import torch
from torchvision.utils import draw_bounding_boxes
import yaml

import xml.sax


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configure information."""
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    config["train"]["log_path"] = args.log_path

    return config


def create_log_path(log_path: str) -> Tuple[str, str]:
    """Create the directory for Tensorboard and checkpoints based on current time."""
    cur_time = datetime.now().strftime("%Y%m%d%H%M%S")
    tensorboard_path = os.path.join(log_path, cur_time, "tensorboard")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    checkpoint_path = os.path.join(log_path, cur_time, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    return tensorboard_path, checkpoint_path


def visualize_image(
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: List[str],
    gt_boxes: torch.Tensor = None,
    gt_labels: List[str] = None,
) -> torch.Tensor:
    """Visualize the image"""
    image_uint8 = (255.0 * image).to(torch.uint8)
    image_uint8 = draw_bounding_boxes(
        image_uint8, pred_boxes, pred_labels, colors="red"
    )

    if gt_boxes is not None and gt_labels is not None:
        image_uint8 = draw_bounding_boxes(
            image_uint8, gt_boxes, gt_labels, colors="green"
        )
    
    # Return (H, W, C).
    return image_uint8.permute(1, 2, 0)


def visual_class_distribution(file_path):
    from xml.dom.minidom import parse
    import os
    g = os.walk(file_path)
    print(g)
    classes = {}

    for path,dir_list,file_list in g:  
        print(path,dir_list)
        for dir_name in file_list:
            # print(dir_name)
            fime_name = os.path.join(path, dir_name)
            print(fime_name)
            # 读取文件
            dom = parse(fime_name)
            # 获取文档元素对象
            data = dom.documentElement
            # 获取 student
            objects = data.getElementsByTagName('object')
            for obj in objects:
                # 获取标签中内容
                name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
                if name in classes.keys():
                    classes[name] += 1
                else:
                    classes[name] = 1
                print('name:', name)
    print(classes)




file_path = "/data/VOCdevkit/VOC2012/Annotations/"
visual_class_distribution(file_path)
