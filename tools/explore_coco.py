import copy
import json
import os.path

import numpy as np
import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO


def main():
    imgs_dir = 'data/SIXRay10/image/test'
    gt_json = '/home/brian/Documents/datasets/SIXRay10/annotation/SIXRay10_test_no_firearm.json'
    results_file = "/home/brian/Documents/Projects/vos/detection/data/SIXRAY10-Detection/faster-rcnn/vos/random_seed_0/inference/sixray10_val_id/standard_nms/corruption_level_0/coco_instances_results.json"

    coco_annotation = COCO(gt_json)
    coco_annotation = coco_annotation.loadRes(results_file)

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    # Category ID -> Category Name.
    query_id = cat_ids[0]
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    print("Category ID -> Category Name:")
    print(
        f"Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}"
    )

    # Category Name -> Category ID.
    # query_name = cat_names[2]
    # query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
    # print("Category Name -> ID:")
    # print(f"Category Name: {query_name}, Category ID: {query_id}")

    # Get the ID of all the images containing the object of the category.
    img_ids = coco_annotation.getImgIds()
    print(f"Number of Images Containing {query_name}: {len(img_ids)}")

    # Pick one image.
    for img_id in img_ids:
        img_info = coco_annotation.loadImgs([img_id])[0]
        img_file_name = os.path.join(imgs_dir, img_info["file_name"])

        # Get all the annotations for the specified image.
        ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco_annotation.loadAnns(ann_ids)
        for r in anns:
            r['ood_score'] = torch.logsumexp(torch.tensor(r['inter_feat'][:-1]), dim=0).sigmoid().item()
        anns = [a for a in anns if a['score'] > 0.2]  # and a['ood_score'] < 0.99966]
        print(f"Annotations for Image ID {img_id}:")
        print(anns)

        # Use URL to load image.
        im = Image.open(img_file_name)

        # Save image and its labeled version.
        plt.figure()
        plt.axis("off")
        plt.imshow(np.asarray(im))
        # Plot segmentation and bounding box.
        coco_annotation.showAnns(anns, draw_bbox=True)
        plt.savefig(f"{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)

    return


if __name__ == "__main__":

    main()