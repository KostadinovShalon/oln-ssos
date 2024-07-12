import argparse
import json
import os
import warnings

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.apis import multi_gpu_test, single_gpu_test, init_detector, inference_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

normal_img_dir = '/home/brian/Documents/datasets/gilardoni_parcels/top-view'
img_dir = '/home/brian/Documents/datasets/gilardoni_parcels/top-view/with_anomalies'
score_thr = 0.4
# models = [
#     'oln_ffs_mask_db6_kmeans_5',
#     'oln_ffs_box_sixray10_kmeans_5',
#     'oln_ffs_box_sixray10_kmeans_10',
#     'oln_ffs_box_sixray10_kmeans_10_ns50',
#     'oln_ffs_box_sixray10_kmeans_10_ns200',
#     'oln_ffs_box_sixray10_kmeans_10_ns300',
#     'oln_ffs_box_sixray10_kmeans_10_s2',
#     'oln_ffs_box_sixray10_kmeans_10_s6',
#     'oln_ffs_box_sixray10_kmeans_20',
#     'oln_vos_box_sixray10_kmeans_5',
#     'oln_vos_box_sixray10_kmeans_10',
#     'oln_vos_box_sixray10_kmeans_10_ns1000',
#     'oln_vos_box_sixray10_kmeans_10_ns5000',
#     'oln_ffs_mask_db6_kmeans_5_nll0.01',
#     'oln_ffs_mask_db6_kmeans_5_nll0.001',
#     'oln_ffs_mask_db6_kmeans_5_ow0.5',
#     'oln_ffs_mask_db6_kmeans_5_ow1',
#     'oln_ffs_mask_db6_kmeans_5_pw0.1',
#     'oln_ffs_mask_db6_kmeans_5_s4',
#     'oln_ffs_mask_db6_kmeans_5_s6',
#     'oln_ffs_mask_db6_kmeans_20',
#     'oln_ffs_mask_db6_kmeans_20_nll0.01',
#     'oln_ffs_mask_db6_kmeans_20_nll0.001',
#     'oln_ffs_mask_db6_kmeans_20_ow0.5',
#     'oln_ffs_mask_db6_kmeans_20_ow1',
#     'oln_ffs_mask_db6_kmeans_20_pw0.1',
#     'oln_ffs_mask_db6_kmeans_20_s4',
#     'oln_ffs_mask_db6_kmeans_20_s6',
#     'oln_ffs_mask_db6_kmeans_100',
#     'oln_ffs_mask_db6_kmeans_100_nll0.01',
#     'oln_ffs_mask_db6_kmeans_100_nll0.001',
#     'oln_ffs_mask_db6_kmeans_100_ow0.5',
#     'oln_ffs_mask_db6_kmeans_100_ow1',
#     'oln_ffs_mask_db6_kmeans_100_pw0.1',
#     'oln_ffs_mask_db6_kmeans_100_s4',
#     'oln_ffs_mask_db6_kmeans_100_s6',
#     'oln_ffs_box_sixray10_kmeans_20_nll0.01',
#     'oln_ffs_box_sixray10_kmeans_20_nll0.001',
#     'oln_ffs_box_sixray10_kmeans_20_nll0.00001',
#     'oln_ffs_box_sixray10_kmeans_20_ns50',
#     'oln_ffs_box_sixray10_kmeans_20_ns200',
#     'oln_ffs_box_sixray10_kmeans_20_ns300',
#     'oln_ffs_box_sixray10_kmeans_20_ns500',
#     'oln_ffs_box_sixray10_kmeans_20_ns1000',
#     'oln_ffs_box_dbf6_kmeans_20_ns50',
#     'oln_ffs_box_dbf6_kmeans_20_ns100',
#     'oln_ffs_box_dbf6_kmeans_20_ns200',
#     'oln_ffs_box_dbf6_kmeans_20_ns300',
#     'oln_ffs_box_dbf6_kmeans_20_ns500',
#     'oln_ffs_box_dbf6_kmeans_20_ns1000',
#     'oln_ffs_mask_dbf6_kmeans_20_ns50',
#     'oln_ffs_mask_dbf6_kmeans_20_ns100',
#     'oln_ffs_mask_dbf6_kmeans_20_ns200',
#     'oln_ffs_mask_dbf6_kmeans_20_ns300',
#     'oln_ffs_mask_dbf6_kmeans_20_ns500',
#     'oln_ffs_mask_dbf6_kmeans_20_ns1000',
# ]
#
# thr = [0.99761,
#        0.99173,
#        0.99442,
#        0.99598,
#        0.99648,
#        0.99668,
#        0.99746,
#        0.99733,
#        0.83977,
#        0.8305,
#        0.83694,
#        0.85765,
#        0.99837,
#        0.99331,
#        0.99597,
#        0.99764,
#        0.99813,
#        0.99471,
#        0.99563,
#        0.9956,
#        0.99661,
#        0.99641,
#        0.995,
#        0.99654,
#        0.99746,
#        0.99823,
#        0.99612,
#        0.99689,
#        0.99112,
#        0.99441,
#        0.99491,
#        0.99237,
#        0.9926,
#        0.99067,
#        0.98856,
#        0.99377,
#        0.99431,
#        0.99459,
#        0.99584,
#        0.99608,
#        0.99631,
#        0.99726,
#        0.99456,
#        0.99683, 0.99599,
#        0.99355,
#        0.99401,
#        0.99296,
#        0.99685,
#        0.99632,
#        0.99643,
#        0.99688,
#        0.99542,
#        0.99725,
#        0.99572,
#        0.99799,
#        ]
# show_bbox = True
models = ['oln_ffs_mask_parcels_kmeans_20']
thr = [0.976]
show_bbox = True

def main():
    for m, t in zip(models, thr):

        db = 'sixray10' if 'sixray10' in m else 'db6'

        cfg = f"configs/parcels/{m}.py"
        checkpoint = f"work_dirs/{m}/epoch_8.pth"
        out_dir = f'/home/brian/Documents/datasets/gilardoni_parcels/{m}/'
        device = "cuda:0"
        anomaly_score_threshold = t
        model = init_detector(cfg, checkpoint, device=device)
        ano_scores = []

        # normal_imgs = os.listdir(normal_img_dir)
        # for img_name in normal_imgs:
        #     if img_name.endswith(('.jpg', '.jpeg', '.png')):
        #         img_path = os.path.join(normal_img_dir, img_name)
        #         result = inference_detector(model, img_path)
        #         if len(result) == 0:
        #             continue
        #         try:
        #             if isinstance(result, tuple):
        #                 result = result[0]
        #             obj_scores = result[0][:, 4]
        #             anomaly_scores = result[0][:, 5]
        #             ano_scores.extend(anomaly_scores[obj_scores > score_thr].tolist())
        #         except:
        #             continue
        #
        # ano_scores.sort()
        # anomaly_score_threshold = ano_scores[int(len(ano_scores) * 0.05)]
        if hasattr(model, 'module'):
            model = model.module
        imgs = os.listdir(img_dir)
        for img_name in imgs:
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, img_name)
                result = inference_detector(model, img_path)
                out_file = os.path.join(out_dir, img_name)
                model.show_result(img_path, result, score_thr=score_thr, show=True,
                                  ood_thr=anomaly_score_threshold,
                                  show_bboxes=show_bbox,
                                  mask_color=(0, 0, 255),
                                  out_file=out_file)


if __name__ == '__main__':
    main()
