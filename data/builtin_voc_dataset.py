import os
import numpy as np

# PASCAL VOC categories
PASCAL_VOC_ALL_CATEGORIES = {
    1: [
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
        "bird",
        "bus",
        "cow",
        "motorbike",
        "sofa",
    ],
    2: [
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
        "aeroplane",
        "bottle",
        "cow",
        "horse",
        "sofa",
    ],
    3: [
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
        "boat",
        "cat",
        "motorbike",
        "sheep",
        "sofa",
    ],
}

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}

PASCAL_VOC_BASE_CATEGORIES = {
    1: [
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    ],
    2: [
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    ],
    3: [
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
    ],
}

image_set = "train"
root = "/data/"
voc_root = os.path.join(root, "VOCdevkit", "VOC")
splits_dir = os.path.join(voc_root, "ImageSets", "Main")
# split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
# with open(os.path.join(split_f)) as f:
#     all_names = [x.strip() for x in f.readlines()]
#     print("************* ", len(all_names))

for sid in range(1, 4):
    for shot in [1, 2, 3, 5, 10]:
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            all_names = [x.strip() for x in f.readlines()]
            print("************* ", len(all_names), all_names[0])
        for cls in PASCAL_VOC_NOVEL_CATEGORIES[sid]:
            split_f = os.path.join(splits_dir, cls + "_train.txt")
            
            with open(os.path.join(split_f)) as f:
                
                few_names = [x.strip() for x in f.readlines() if not x.strip().endswith('-1')]
                # print(few_names)
            
            slice = np.random.choice(len(few_names), len(few_names) - shot)
            # print(slice)
            few_names = [few_names[i].split(' ')[0] for i in slice]
            print("shot = {}, len(few_names) = {}".format(shot, len(few_names)))

            all_names = [i for i in all_names if i not in few_names]
            print("aaaaa********* ", len(all_names))
        with open(os.path.join(splits_dir, "train_{}shot_{}".format(shot, sid) + ".txt"), 'a') as fp:
            print(os.path.join(splits_dir, "train_{}shot_{}".format(shot, sid) + ".txt"))
            [fp.write(str(item)+'\n') for  item in all_names]
            fp.close()
    break
               

        

            

        
            
        

