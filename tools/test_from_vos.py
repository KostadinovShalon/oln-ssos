import itertools
import json
import os.path
import sys

import cv2
import numpy as np
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from matplotlib import pyplot as plt
from terminaltables import AsciiTable

# LTDImaging
from mmdet.core.visualization import color_val_matplotlib

# id_results_path = '/home/brian/Documents/Projects/siren/rcnn/detection/data/LTDIMAGING-Detection/faster-rcnn/center64_0.1/random_seed_0/inference/ltdimaging_val_id/standard_nms/corruption_level_0/coco_instances_results.json'
# ood_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/LTDIMAGING-Detection/faster-rcnn/regnetx/random_seed_0/inference/ltdimaging_val_ood/standard_nms/corruption_level_0/coco_instances_results.json'
# id_test_path = '/home/brian/Documents/Projects/oln-vos/data/ltdimaging/test_day_no_vehicles.json'
# ood_test_path = '/home/brian/Documents/Projects/oln-vos/data/ltdimaging/test_week_only_vehicles.json'
# score_thr = 0.5
# ood_score = 0.99634
# ood_imgs_dir = 'data/ltdimaging/Week/images/'
# out_dir = 'qualitative/ltdimaging_ffs'
# os.makedirs(out_dir, exist_ok=True)

# SIXRay10
# id_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/SIXRAY10-Detection/faster-rcnn/regnetx/random_seed_0/inference/sixray10_val_id/standard_nms/corruption_level_0/coco_instances_results.json'
# ood_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/SIXRAY10-Detection/faster-rcnn/regnetx/random_seed_0/inference/sixray10_val_ood/standard_nms/corruption_level_0/coco_instances_results.json'
# score_thr = 0.6
# ood_score = 0.99414
# id_test_path = '/home/brian/Documents/datasets/SIXRay10/annotation/SIXRay10_test_no_firearm.json'
# ood_test_path = '/home/brian/Documents/datasets/SIXRay10/annotation/SIXRay10_test_only_firearms.json'
# ood_imgs_dir = 'data/SIXRay10/image/test/'
# out_dir = 'qualitative/sixray10_ffs'
# os.makedirs(out_dir, exist_ok=True)

# COCO
id_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/inference/voc_custom_val/standard_nms/corruption_level_0/coco_instances_results.json'
ood_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/VOC-Detection/faster-rcnn/regnetx/random_seed_0/inference/coco_ood_val/standard_nms/corruption_level_0/coco_instances_results.json'
id_test_path = '/home/brian/Documents/datasets/VOC/VOC_0712_converted/val_coco_format.json'
ood_test_path = '/home/brian/Documents/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json'
score_thr = 0.6
ood_score = 0.999998
ood_imgs_dir = 'data/coco/val2017/'
out_dir = 'qualitative/voc_coco_ffs'
os.makedirs(out_dir, exist_ok=True)
# ood_results_path = '/home/brian/Documents/Projects/vos/detection/data/VOC-Detection/faster-rcnn/vos/random_seed_0/inference/openimages_ood_val/standard_nms/corruption_level_0/coco_instances_results.json'
# ood_test_path = '/home/brian/Documents/datasets/OpenImages_vos/OpenImages/ood_classes_rm_overlap/COCO-Format/val_coco_format_with_bboxes.json'

# BDD
# id_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/BDD100k/faster-rcnn/regnet/random_seed_0/inference/bdd_custom_val/standard_nms/corruption_level_0/coco_instances_results.json'
# ood_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/BDD100k/faster-rcnn/regnet/random_seed_0/inference/coco_ood_val_bdd/standard_nms/corruption_level_0/coco_instances_results.json'

# id_results_path = '/home/brian/Documents/Projects/oln-vos/work_dirs/oln_ffs_box_bdd_kmeans_10/epoch_6.id.bbox.json'
# ood_results_path = '/home/brian/Documents/Projects/oln-vos/work_dirs/oln_ffs_box_bdd_kmeans_10/epoch_6.ood.bbox.json'
# id_test_path = '/home/brian/Documents/Projects/oln-vos/data/bdd100k/val_bdd_converted.json'
# ood_test_path = '/home/brian/Documents/Projects/oln-vos/data/coco/annotations/instances_val2017_ood_wrt_bdd_rm_overlap.json'
# ood_score = 0.98444
# ood_imgs_dir = 'data/coco/val2017/'
# out_dir = 'qualitative/bdd_coco_ffs'
# os.makedirs(out_dir, exist_ok=True)
# score_thr = 0.2

# DB6
# id_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/DB6-Detection/faster-rcnn/regnetx/random_seed_0/inference/db6_val_id/standard_nms/corruption_level_0/coco_instances_results.json'
# ood_results_path = '/home/brian/Documents/Projects/FFS/Flow-Feature-Synthesis/detection/data/DB6-Detection/faster-rcnn/regnetx/random_seed_0/inference/db6_val_ood/standard_nms/corruption_level_0/coco_instances_results.json'
# id_test_path = '/home/brian/Documents/Projects/oln-vos/data/db6/annotations/test_db6_no_firearms.json'
# ood_test_path = '/home/brian/Documents/Projects/oln-vos/data/db6/annotations/test_db6_only_firearms.json'
# ood_imgs_dir = 'data/db6/images/'
# ood_score = 0.99463
# score_thr = 0.6
# out_dir = 'qualitative/db6_ffs'
# os.makedirs(out_dir, exist_ok=True)
# score_thr = 0.6498

# # ood_score = 0.9881  # ltd
# ood_score = 0.99935  # sixray
# ood_score = 0.999951
id_results_json = json.load(open(id_results_path, 'rb'))
ood_results_json = json.load(open(ood_results_path, 'rb'))

if 'ood_score' not in id_results_json[0].keys():
    for r in id_results_json:
        r['ood_score'] = torch.logsumexp(torch.tensor(r['inter_feat'][:-1]), dim=0).sigmoid().item()

    for r in ood_results_json:
        r['ood_score'] = torch.logsumexp(torch.tensor(r['inter_feat'][:-1]), dim=0).sigmoid().item()
        r['category_id'] = 1

id_gt_coco = COCO(id_test_path)
ood_gt_coco = COCO(ood_test_path)
id_res_coco = id_gt_coco.loadRes(id_results_json)
ood_res_coco = ood_gt_coco.loadRes(ood_results_json)

# for i in list(id_res_coco.imgs.keys())[:10]:
#     im = id_res_coco.imgs[i]
#     anns = id_res_coco.loadAnns(id_res_coco.get_ann_ids(img_ids=im['id']))
#     img = cv2.imread(os.path.join(id_imgs_dir, im['file_name']))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     for ann in anns:
#         if ann['score'] < score_thr or ann['ood_score'] < ood_score:
#             continue
#         p1 = int(ann['bbox'][0]), int(ann['bbox'][1])
#         p2 = int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3])
#         img = cv2.rectangle(img, p1, p2, color=(255, 0, 0), thickness=1)
#     plt.imshow(img)
#     plt.show()
font_size = 30
text_color = (72, 101, 241)
text_color = color_val_matplotlib(text_color)
for i in tqdm.tqdm(list(ood_res_coco.imgs.keys())):
    im = ood_res_coco.imgs[i]
    anns = ood_res_coco.loadAnns(ood_res_coco.get_ann_ids(img_ids=im['id']))
    img = cv2.imread(os.path.join(ood_imgs_dir, im['file_name']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure('', figsize=(15, 10))
    plt.title('')
    plt.axis('off')
    ax = plt.gca()
    for ann in anns:
        if ann['score'] < score_thr or ann['ood_score'] > ood_score:
            continue
        label_text = f"{ann['score']:.02f}|{ann['ood_score']:.07f}"
        ax.text(
            int(ann['bbox'][0]),
            int(ann['bbox'][1]),
            label_text,
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        p1 = int(ann['bbox'][0]), int(ann['bbox'][1])
        p2 = int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3])
        img = cv2.rectangle(img, p1, p2, color=(255, 0, 0), thickness=5)
    plt.imshow(img)
    plt.savefig(os.path.join(out_dir, im['file_name']))
    plt.close()

sys.exit(0)
id_coco_eval = COCOeval(id_gt_coco, id_res_coco, iouType='bbox')
id_coco_eval.params.useCats = 0
id_coco_eval.evaluate()
id_coco_eval.accumulate()
id_coco_eval.summarize()

id_anomaly_scores = [r['ood_score'] for r in id_results_json]
dt_ids_with_match = []
for ev_im in id_coco_eval.evalImgs:
    if ev_im is not None:
        for dt_id in ev_im['gtMatches'][0]:
            if dt_id > 0:
                dt_ids_with_match.append(int(dt_id))
dt_ids_with_match = list(set(dt_ids_with_match))
valid_detections = id_coco_eval.cocoDt.loadAnns(dt_ids_with_match)
optimal_detections = [v for v in valid_detections if v['score'] > score_thr]
ood_scores = [o['ood_score'] for o in optimal_detections]
ood_scores.sort()
anomaly_score_threshold = ood_scores[int(len(ood_scores) * 0.05)]
print('Anomaly threshold:', anomaly_score_threshold)

ood_results_json = [r for r in ood_results_json if r['ood_score'] < anomaly_score_threshold]

ood_res_coco = ood_gt_coco.loadRes(ood_results_json)
ood_coco_eval = COCOeval(ood_gt_coco, ood_res_coco, iouType='bbox')
ood_coco_eval.params.useCats = 0
ood_coco_eval.evaluate()
ood_coco_eval.accumulate()
ood_coco_eval.summarize()
