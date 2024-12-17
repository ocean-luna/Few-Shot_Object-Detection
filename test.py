from pprint import pprint

import matplotlib.pyplot as plt
import torch
from torchmetrics.detection import MeanAveragePrecision

from data.build_dataset import build_dataset, collate_fn
from data.dataset_classes import VOC_DICT
from models.build_model import build_model
from utils.utils import visualize_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs = {
    "dataset": {"path": "/home/liupei/data", "name": "PeSOTIF"},
    "model": {"name": "faster_rcnn", "freeze_rpn": True, "num_classes": 20, "freeze_backbone": True, "cosine_similarity": False},
    "train": {"crop_size": 256},
    "val": {"crop_size": -1},
}

# Initialize the dataloader.
dataset = build_dataset(configs, "train")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
dataloader_iter = iter(dataloader)

model = build_model(configs).to(device)
model.load_state_dict(torch.load('/home/liupei/code/few_shot_learning-main/runs/20240115182844/checkpoints/ckpt_best.pth')['model'])

for name, param in model.named_parameters():
    print("{}:".format(name), param.requires_grad)

images = next(dataloader_iter)
# print(images)
# img = images['image'][0]
# print(img)

for i in range(len(images['image'])):
    images['image'][i] = images['image'][i].transpose(2,0,1)

images = list(img.transpose((2, 0, 1)).to(device) for img in images['image'])
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

model.eval()

with torch.no_grad():
    predictions = model(images)

# Visualize predictions.
pred_boxes = predictions[0]["boxes"].cpu()
pred_labels = predictions[0]["labels"].cpu()
pred_scores = predictions[0]["scores"].cpu()

filtered_boxes = pred_boxes[pred_scores > 0.1]
filtered_labels = pred_labels[pred_scores > 0.1]

image_vis = visualize_image(images[0].cpu(), filtered_boxes, filtered_labels).numpy()

plt.figure(figsize=(5, 5))
plt.imshow(image_vis)
plt.axis("off")
plt.show()