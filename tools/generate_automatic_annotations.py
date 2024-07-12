import json
import os
from itertools import groupby

import cv2
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from pycocotools import mask as cocoMask

from mmdet.apis import init_detector, inference_detector

data = {"images": [], "annotations": [], "categories": [{"supercategory": "object", "name": "object", "id": 1}]}

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def show_anns(masks, opacity=0.75):
    if len(masks) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    sorted_anns = sorted(masks, key=(lambda x: torch.sum(x).cpu().item()), reverse=True)
    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    img[:, :, 3] = 0
    for i, ann in enumerate(sorted_anns):
        m = (ann.cpu().numpy() == 1) if isinstance(ann, torch.Tensor) else ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [opacity]])
        img[m] = color_mask
    ax.imshow(img)


def show_boxes(bboxes, imw, imh):
    img = np.ones((imh, imw, 4))
    img[:, :, 3] = 0
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for bbox in bboxes:
        b = bbox.cpu().numpy()
        tl = int(b[0]), int(b[1])
        br = int(b[2]), int(b[3])
        img = cv2.rectangle(img, tl, br, (1.0, 0, 0, 1.0), 2)
    ax.imshow(img)


device = "cuda:0"
score_thr = 0.4

db = 'db6'
m = 'oln_ffs_mask_db6_kmeans_100_s4'

cfg = f"configs/oln_ffs/{db}/{m}.py"
checkpoint = f"work_dirs/{m}/epoch_8.pth"
model = init_detector(cfg, checkpoint, device=device)

img_dir = '/home/brian/Documents/datasets/gilardoni_parcels/top-view/'
imgs = os.listdir(img_dir)
img_id = 1
ann_id = 1
with torch.no_grad():
    for img_name in tqdm.tqdm(imgs):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img = os.path.join(img_dir, img_name)
            _img = cv2.imread(img)
            result = inference_detector(model, img)
            if len(result) == 0:
                continue
            mask = None
            if isinstance(result, tuple):
                mask = result[1]
                result = result[0]
            masks = mask[0]
            bboxes = result[0][:, :4]
            obj_scores = result[0][:, 4]
            bboxes = bboxes[obj_scores > score_thr]
            masks = [m for i, m in enumerate(masks) if obj_scores[i] > score_thr]

            data['images'].append({
                "file_name": img_name,
                "width": _img.shape[1],
                "height": _img.shape[0],
                "id": img_id
            })

            decoded_masks = [binary_mask_to_rle(m) for m in masks]

            for bbox, mask in zip(bboxes, masks):
                b = bbox.tolist()
                a = int(b[2] - b[0]) * int(b[3] - b[1])

                ii, cc = cv2.findContours(np.array(mask, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                segm = ii[0].ravel().tolist()
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "area": a,
                    "iscrowd": 0,
                    "bbox": [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])],
                    "segmentation": [segm]
                }
                # ii, cc = cv2.findContours(np.array(mask, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #
                # ann["segmentation"] = cocoMask.frPyObjects(
                #     ann["segmentation"],
                #     ann["segmentation"]["size"][0],
                #     ann["segmentation"]["size"][1],
                # )

                data['annotations'].append(ann)
                ann_id += 1

            # plt.figure(figsize=(20, 20))
            # plt.imshow(img)
            # show_anns(masks, opacity=.8)
            # plt.axis('off')
            # if boxes is not None:
            #     show_boxes(boxes, img.shape[1], img.shape[0])
            # # plt.savefig(f"fig_{args.mode}_{i}.png")
            # plt.show()
            # plt.close()
            img_id += 1
json.dump(data, open("/home/brian/Documents/datasets/gilardoni_parcels/top_view_normal_data_from_oln.json", "w"))
