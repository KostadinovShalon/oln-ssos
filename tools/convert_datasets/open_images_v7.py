import argparse
import os

import pandas as pd
import tqdm
from PIL import Image
import json
import numpy as np
from skimage import measure
from pycocotools import mask
import cv2


tqdm.tqdm.pandas()

def mask_to_polygon(path):
    """ Taken from https://github.com/cocodataset/cocoapi/issues/131 """
    binary_mask = cv2.imread(path)
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    binary_mask = (binary_mask / 255).astype(np.uint8)
    fortran_ground_truth_binary_mask = np.asfortranarray(binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(binary_mask, 0.5)
    segm = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segm.append(segmentation)
    return segm, ground_truth_bounding_box, ground_truth_area


info = {
    'year': 2022,
    'version': 7,
    'description': 'OpenImages datasets',
    "url": "https://storage.googleapis.com/openimages/web/index.html",
    "date_created": "10-10-2023"
}

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', help='Root dir with annotation files')
parser.add_argument('--partition', default='train')
parser.add_argument('--preloaded_images_json')
parser.add_argument('--preloaded_bboxes_json')
parser.add_argument('--mode', choices=['box', 'segm'], default='segm')
args = parser.parse_args()

splits = ['train', 'test', 'validation']
label_files = ['train-images-boxable-with-rotation.csv', 'test-images-with-rotation.csv',
               'validation-images-with-rotation.csv']
boxes_files = ['oidv6-train-annotations-bbox.csv', 'test-annotations-bbox.csv', 'validation-annotations-bbox.csv']
segm_files = ['train-annotations-object-segmentation.csv', 'test-annotations-object-segmentation.csv',
              'validation-annotations-object-segmentation.csv']

categoires_path = os.path.join(args.root_dir, 'annotations', 'oidv7-class-descriptions-boxable.csv')
segm_categories = os.path.join(args.root_dir, 'annotations', 'oidv7-classes-segmentation.txt')
segm_categories = pd.read_fwf(open(segm_categories, 'r'), header=None).values.flatten().tolist()
label_file_paths = {split: os.path.join(args.root_dir, 'annotations', filename) for split, filename in zip(splits, label_files)}
boxes_file_paths = {split: os.path.join(args.root_dir, 'annotations', filename) for split, filename in zip(splits, boxes_files)}
segm_file_paths = {split: os.path.join(args.root_dir, 'annotations', filename) for split, filename in zip(splits, segm_files)}
p = args.partition


img_counter = 1
ann_counter = 1
images = []
preloaded_file = args.preloaded_images_json
preloaded_bboxes_json = args.preloaded_bboxes_json
data = {}
if preloaded_file is not None:
    data = json.load(open(preloaded_file, 'r'))

if 'info' not in data.keys():
    data['info'] = info

if 'images' not in data.keys():
    image_labels = pd.read_csv(label_file_paths[p])
    for index, img in tqdm.tqdm(image_labels.iterrows(), desc='Reading Images File', total=len(image_labels)):
        img_id = img['ImageID']
        if p == 'train':
            first_letter = img_id[0]
            img_path = os.path.join(args.root_dir, 'images', p, f'{p}_{first_letter}', f'{img_id}.jpg')
        else:
            img_path = os.path.join(args.root_dir, 'images', p, f'{img_id}.jpg')
        _img = Image.open(img_path)
        w, h = _img.size
        image = {
            "id": img_counter,
            "width": w,
            "height": h,
            "file_name": f'{img_id}.jpg'
        }
        images.append(image)
        img_counter += 1
    data['images'] = images
    json.dump(data, open(os.path.join(args.root_dir, 'annotations', f'images_{p}_first_stage.json'), 'w'))


categories_df = pd.read_csv(categoires_path)
cats_oi_dict = {}
cats_list = []

cat_counter = 1
for i, cat in categories_df.iterrows():
    if args.mode == 'segm':
        if cat['LabelName'] not in segm_categories:
            continue
    cats_oi_dict[cat['LabelName']] = cat_counter
    cats_list.append({
        'id': cat_counter,
        'name': cat['DisplayName']
    })
    cat_counter += 1

images_id_dict = {}

for img in tqdm.tqdm(data['images']):
    img_id = img['file_name']
    img_id = img_id.rsplit('.', 1)[0]
    images_id_dict[img_id] = img['id'], img['width'], img['height']

data['categories'] = cats_list
anns = []
if args.mode == 'box':
    if preloaded_bboxes_json is not None:
        print('Loading Annotations')
        anns = json.load(open(preloaded_bboxes_json, 'r'))
    else:
        print("Reading annotations")
        image_boxes_anns = pd.read_csv(boxes_file_paths[p])
        image_numeric_ids = image_boxes_anns['ImageID'].progress_map(lambda x: images_id_dict[x][0])
        image_widths = image_boxes_anns['ImageID'].progress_map(lambda x: images_id_dict[x][1])
        image_heights = image_boxes_anns['ImageID'].progress_map(lambda x: images_id_dict[x][2])
        x1 = image_boxes_anns['XMin'] * image_widths
        x2 = image_boxes_anns['XMax'] * image_widths
        y1 = image_boxes_anns['YMin'] * image_heights
        y2 = image_boxes_anns['YMax'] * image_heights
        w = x2 - x1
        h = y2 - y1
        area = w * h
        category_ids = image_boxes_anns['LabelName'].progress_map(lambda x: cats_oi_dict[x])
        anns = pd.DataFrame({
            "id": pd.Series(range(1, len(image_boxes_anns) + 1)),
            "image_id": image_numeric_ids,
            "area": area,
            "iscrowd": image_boxes_anns['IsGroupOf'],
            "category_id": category_ids,
            "x1": x1,
            "y1": y1,
            "w": w,
            "h": h,
        })
        bboxes_json_path = os.path.join(args.root_dir, 'annotations', f'bboxes_{p}_first_stage.json')
        print("Writing bboxes json, first stage")
        anns.to_json(bboxes_json_path, orient='records')
        del image_boxes_anns, image_widths, image_numeric_ids, image_heights, x1, x2, y1, y2, w, h, category_ids, anns, area
        print("Loading annotations after freeing-up memory")
        anns = json.load(open(bboxes_json_path, 'r'))
    for ann in tqdm.tqdm(anns, desc="Formatting BBoxes"):
        ann['bbox'] = [ann['x1'], ann['y1'], ann['w'], ann['h']]
        del ann['x1'], ann['y1'], ann['w'], ann['h']
else:
    image_segm_anns = pd.read_csv(segm_file_paths[p])
    ids = pd.Series(range(1, len(image_segm_anns) + 1))
    image_numeric_ids = image_segm_anns['ImageID'].progress_map(lambda x: images_id_dict[x][0])
    if p == 'train':
        mask_paths = image_segm_anns['MaskPath'].progress_map(
            lambda x: os.path.join(args.root_dir, 'masks', p, f'train-masks-{x[0]}', x)
        )
    else:
        mask_paths = image_segm_anns['MaskPath'].progress_map(
            lambda x: os.path.join(args.root_dir, 'masks', p, x)
        )
    category_ids = image_segm_anns['LabelName'].progress_map(lambda x: cats_oi_dict[x])
    polygons = mask_paths.progress_map(mask_to_polygon)
    segms = polygons.map(lambda x: x[0])
    bboxes = polygons.map(lambda x: x[1])
    areas = polygons.map(lambda x: x[2])

    anns = pd.DataFrame({
        "id": ids,
        "image_id": image_numeric_ids,
        "area": areas,
        "category_id": category_ids,
        "bbox": bboxes,
        "segm": segms
    })
    instances_json_path = os.path.join(args.root_dir, 'annotations', f'instances_{p}_first_stage.json')
    anns.to_json(instances_json_path, orient='records')

    # for counter, seg in tqdm.tqdm(image_segm_anns.iterrows(), total=len(image_segm_anns)):
    #     im_id, _, _ = images_id_dict[seg['ImageID']]
    #     mask_path = seg['MaskPath']
    #     if p == 'train':
    #         first_letter = mask_path[0]
    #         mask_path = os.path.join(args.root_dir, 'masks', p, f'train-masks-{first_letter}', mask_path)
    #     else:
    #         mask_path = os.path.join(args.root_dir, 'masks', p, mask_path)
    #     s, b, area = mask_to_polygon(mask_path)
    #     ann = {
    #         "id": ann_counter,
    #         "image_id": im_id,
    #         "area": area,
    #         "bbox": b,
    #         "iscrowd": 0,
    #         'segm': s,
    #         'category_id': cats_oi_dict[seg['LabelName']]
    #     }
    #     ann_counter += 1
    #     anns.append(ann)

if 'images' not in data.keys():
    data['images'] = images
data['annotations'] = anns
name = f'instances_{p}.json' if args.mode == 'segm' else f'detection_{p}.json'
out_file = os.path.join(args.root_dir, name)
print(f"Writing resulting coco file to: {out_file}")
json.dump(data, open(out_file, 'w'))

