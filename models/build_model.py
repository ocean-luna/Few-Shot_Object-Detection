from typing import Any, Dict

import torch

from models.faster_rcnn import FewShotFasterRCNN

MODEL_NAMES = ["faster_rcnn"]


def build_model(configs: Dict[str, Any]) -> torch.nn.Module:
    """Build the model based on the input specs."""
    model_name = configs["model"]["name"]

    if model_name not in MODEL_NAMES:
        raise ValueError("Unsupported model: {}.".format(model_name))

    if model_name == "faster_rcnn":
        model = FewShotFasterRCNN(
            num_classes=configs["model"]["num_classes"],
            freeze_backbone=configs["model"]["freeze_backbone"],
            freeze_rpn=configs["model"]["freeze_rpn"],
            cosine_similarity=configs["model"]["cosine_similarity"],
        )

    return model
