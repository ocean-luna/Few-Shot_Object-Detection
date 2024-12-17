"""Model training.

Example command line:
python train.py \
    --log_path=runs/exp3
"""
import argparse
import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.metric import Metric

from data.build_dataset import build_dataset, collate_fn
from data.dataset_classes import VOC_DICT
from models.build_model import build_model
from utils.utils import load_config, create_log_path, visualize_image


def args_parser() -> argparse.Namespace:
    """Parse input parameters."""
    parser = argparse.ArgumentParser(description="Few-shot model training.")
    parser.add_argument(
        "--log_path",
        type=str,
        default="runs",
        help="The logs file path.",
        # required=True,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/basic.yaml",
        help="The configure file path.",
    )
    args = parser.parse_args()

    return args


def train(configs: Dict[str, Any]) -> None:
    """The main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the detection model.
    model = build_model(configs)

    # Initialize the training and validation dataloaders.
    train_dataset = build_dataset(configs, dataset_type="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_dataset = build_dataset(configs, dataset_type="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size"],
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Initialize the optimizer to update specific weights.
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_params, lr=configs["train"]["init_lr"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=configs["train"]["lr_scheduler_step_size"],
        gamma=configs["train"]["lr_scheduler_gamma"],
    )

    # Initialize the tensorboard.
    tensorboard_path, checkpoint_path = create_log_path(configs["train"]["log_path"])
    log_writer = SummaryWriter(tensorboard_path)

    # Initialize the evaluation metric.
    eval_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    # Load the checkpoint if specified.
    start_epoch = 0
    iter_number = 0
    if configs["train"]["pretrained_checkpoint_path"]:
        checkpoint = torch.load(
            configs["train"]["pretrained_checkpoint_path"], map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]

    # Transfer the model to the target device.
    model.to(device)

    best_map50 = 0

    # Start the model training.
    for epoch in range(start_epoch, configs["train"]["total_epoch"]):
        print("Start epoch {} training.".format(epoch))

        model.train()
        mean_losses = 0
        for images, targets in train_dataloader:
            optimizer.zero_grad()

            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            mean_losses += losses

            if (iter_number + 1) % configs["train"]["log_frequency"] == 0:
                for loss_name in loss_dict:
                    log_writer.add_scalar(
                        "Train/{}".format(loss_name), loss_dict[loss_name], iter_number
                    )
                log_writer.add_scalar(
                    "Train/total_loss",
                    mean_losses / configs["train"]["log_frequency"],
                    iter_number,
                )
                log_writer.add_scalar(
                    "Train/learning_rate", lr_scheduler.get_last_lr()[0], iter_number
                )
                mean_losses = 0

            losses.backward()
            optimizer.step()

            iter_number += 1

        # Run validation.
        val_results, visual_images = validate_epoch(
            model, val_dataloader, eval_metric, device
        )

        # # Log valiadtion results.
        # log_writer.add_scalar("Val/map", val_results["map"], iter_number - 1)
        # log_writer.add_scalar("Val/map_50", val_results["map_50"], iter_number - 1)
        # log_writer.add_scalar("Val/map_75", val_results["map_75"], iter_number - 1)
        # for image_id, visual_image in enumerate(visual_images):
        #     log_writer.add_image(
        #         "Val/images_{:02d}".format(image_id),
        #         visual_image,
        #         iter_number - 1,
        #         dataformats="HWC",
        #     )

        lr_scheduler.step()

        # Save the checkpoint.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(checkpoint_path, "ckpt_{:05d}.pth".format(iter_number)),
        )

        if val_results["map_50"] > best_map50:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(checkpoint_path, "ckpt_best.pth"),
            )
            best_map50 = val_results["map_50"]
        print('finish epoch {}'.format(epoch))


def validate_epoch(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    eval_metric: Metric,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
    """The validation function at the end of each epoch."""
    model.eval()
    

    visual_images = []
    with torch.no_grad():
        for batch_id, (images, targets) in enumerate(val_dataloader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)
            eval_metric.update(predictions, targets)

            # Plot prediction results for first 4 validation images.
            if batch_id < 4:
                # Add the visualization image.
                image = images[0].cpu()
                pred_scores = predictions[0]["scores"].cpu()
                pred_boxes = predictions[0]["boxes"].cpu()[pred_scores > 0.5]
                pred_labels = [
                    list(VOC_DICT.keys())[int(idx)]
                    for idx in predictions[0]["labels"].cpu()[pred_scores > 0.5]
                ]

                gt_boxes = targets[0]["boxes"].cpu()
                gt_labels = [
                    list(VOC_DICT.keys())[int(idx)]
                    for idx in targets[0]["labels"].cpu()
                ]

                visual_images.append(
                    visualize_image(image, pred_boxes, pred_labels, gt_boxes, gt_labels)
                )

    val_results = eval_metric.compute()
    eval_metric.reset()

    return val_results, visual_images


if __name__ == "__main__":
    input_args = args_parser()
    input_configs = load_config(input_args)

    train(input_configs)
