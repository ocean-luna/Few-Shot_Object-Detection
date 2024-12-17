import pickle
from torchmetrics.detection.mean_ap import MeanAveragePrecision
f_read = open('/data/test/VOCdevkit/save_fig/dict_file.pkl', 'rb')
dict2 = pickle.load(f_read)
f_read.close()

predictions = dict2['predictions']
targets = dict2['targets']
eval_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

for i in range(len(predictions)):
    eval_metric.update(predictions[i], targets[i])

    print(targets[i][0]["labels"].cpu())

val_results = eval_metric.compute()
eval_metric.reset()



print("map = {},  map_50 = {}".format(val_results["map"], val_results["map_50"]))