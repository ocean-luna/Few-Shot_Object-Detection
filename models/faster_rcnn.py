from typing import Dict, List, Optional, Tuple

import torch
import torchvision


class FewShotFasterRCNN(torch.nn.Module):
    """Implementation of Faster R-CNN for Few-shot learning."""

    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        freeze_rpn: bool = True,
        cosine_similarity: bool = False,
    ) -> None:
        super().__init__()

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        )
        box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                1024, num_classes
            )
            if not cosine_similarity
            else CosineSimilarityFastRCNNPredictor(1024, num_classes)
        )
        self.model.roi_heads.box_predictor = box_predictor

        # Freeze backbone if specified.
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        if freeze_rpn:
            for param in self.model.rpn.parameters():
                param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        return self.model(x, targets)


class CosineSimilarityFastRCNNPredictor(torch.nn.Module):
    """Customized predictor using the Cosine Similarity loss.

    Reference: https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/modeling/roi_heads/fast_rcnn.py#L403
    """

    def __init__(
        self, in_channels: int, num_classes: int, cosine_scale: int = 20
    ) -> None:
        super().__init__()

        self.cls_score = torch.nn.Linear(in_channels, num_classes, bias=False)
        self.cosine_scale = torch.nn.Parameter(torch.ones(1) * cosine_scale)
        self.bbox_pred = torch.nn.Linear(in_channels, num_classes * 4)

        # Prediction head weight initialization.
        torch.nn.init.normal_(self.cls_score.weight, std=0.01)
        torch.nn.init.normal_(self.bbox_pred.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_pred.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        x_norm = x.div(torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x) + 1e-5)

        cls_score_weight_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=-1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            cls_score_weight_norm + 1e-5
        )
        cos_dist = self.cls_score(x_norm)
        scores = self.cosine_scale * cos_dist

        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
