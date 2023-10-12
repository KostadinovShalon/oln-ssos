import json
import numpy as np
import os
import cv2


root_dir = '/home/brian/Documents/datasets/dataset_AnomalyTrack/images'
out_dir = '/home/brian/Documents/datasets/dataset_AnomalyTrack/'


os.makedirs(out_dir, exist_ok=True)

frames = os.listdir(root_dir)

coco = {"images": list(), "categories": [{'id': 1, 'name': 'anomaly'}], 'annotations': list()}

ann_id = 1
for k, frame in enumerate(frames):
    frame_path = os.path.join(root_dir, frame)
    img = cv2.imread(frame_path)

    coco['images'].append({
        'id': k + 1,
        'width': img.shape[1],
        'height': img.shape[0],
        'file_name': frame,
    })
    cv2.imwrite(os.path.join(out_dir, frame), img)

json.dump(coco, open(os.path.join(out_dir, 'instances.json'), 'w'))



