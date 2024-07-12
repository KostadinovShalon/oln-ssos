import json
import pandas as pd


oi_coco_file = json.load(open("/home/brian/Documents/datasets/OpenImages_vos/OpenImages/ood_classes_rm_overlap/COCO-Format/val_coco_format.json",
                              "rb"))

oi_original_file = pd.read_csv("/home/brian/Documents/datasets/OpenImages_vos/OpenImages/ood_classes_rm_overlap/train-annotations-bbox.csv")
print("HI")