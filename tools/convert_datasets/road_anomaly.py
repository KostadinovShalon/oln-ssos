import json
import numpy as np
import os
import cv2


root_dir = '/home/brian/Documents/datasets/RoadAnomaly_jpg/frames'
frame_list = '/home/brian/Documents/datasets/RoadAnomaly_jpg/frame_list.json'
out_dir = '/home/brian/Documents/datasets/RoadAnomaly_jpg/drawn'

os.makedirs(out_dir, exist_ok=True)

frames = json.load(open(frame_list, 'rb'))
files_with_no_index = []

coco = {"images": list(), "categories": [{'id': 1, 'name': 'anomaly'}], 'annotations': list()}

ann_id = 1
for k, frame in enumerate(frames):
    labels_dir = frame.rsplit('.', 1)[0]
    labels_dir = os.path.join(root_dir, f"{labels_dir}.labels")
    frame_path = os.path.join(root_dir, frame)
    img = cv2.imread(frame_path)

    seg_file = os.path.join(f"{labels_dir}.labels", "labels_semantic.png")

    if os.path.exists(os.path.join(labels_dir, 'index.json')):
        index = json.load(open(os.path.join(labels_dir, 'index.json'), 'rb'))
        for instance in index['instances']:
            cv2.rectangle(img, tuple(instance["roi_rect"][0]), tuple(instance["roi_rect"][1]), (0, 0, 255), 3)
            coco['annotations'].append({
                'id': ann_id,
                'image_id': k + 1,
                'category_id': 1,
                'iscrowd': 0,
                'area': float((instance["roi_rect"][1][0] - instance["roi_rect"][0][0]) *
                        (instance["roi_rect"][1][1] - instance["roi_rect"][0][1])),
                'bbox': [instance["roi_rect"][0][0],
                         instance["roi_rect"][0][1],
                         instance["roi_rect"][1][0] - instance["roi_rect"][0][0],
                         instance["roi_rect"][1][1] - instance["roi_rect"][0][1]]
            })
            ann_id += 1
    else:
        semantic = cv2.imread(os.path.join(labels_dir, "labels_semantic.png"))[..., 0]
        semantic = np.array(semantic == 2, dtype=np.uint8)
        output = cv2.connectedComponentsWithStats(
            semantic, 8, cv2.CV_32S)
        numLabels, labels, stats, centroids = output

        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 64:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': k + 1,
                    'category_id': 1,
                    'iscrowd': 0,
                    'area': float(area),
                    'bbox': [int(x), int(y), int(w), int(h)]
                })
                ann_id += 1
        files_with_no_index.append(labels_dir)
    coco['images'].append({
        'id': k + 1,
        'width': img.shape[1],
        'height': img.shape[0],
        'file_name': frame,
        'seg_file_name': seg_file
    })
    cv2.imwrite(os.path.join(out_dir, frame), img)
print("Files with no index:")
for f in files_with_no_index:
    print(f)

json.dump(coco, open(os.path.join(out_dir, 'instances.json'), 'w'))



