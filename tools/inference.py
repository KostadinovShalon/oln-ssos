#!/usr/bin/env python
##########################################################################
# Example :
##########################################################################

import argparse
import json
import os
import warnings

# import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
# from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
#                          wrap_fp16_model)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.apis import multi_gpu_test, single_gpu_test, init_detector, inference_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
##########################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('cfg', help='test config file path for the in-distribution dataset')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thr', type=float, help='bbox score threshold', default=0.3)
    parser.add_argument('--show-bbox', action='store_true', help='show bbox results')
    parser.add_argument('--show-dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--img-dir', help='dir with images')
    parser.add_argument('--anomaly-threshold', type=float, default=0.991)
    args = parser.parse_args()
    return args
##########################################################################


def main():
    args = parse_args()
    anomaly_score_threshold = args.anomaly_threshold
    model = init_detector(args.cfg, args.checkpoint, device=args.device)
    if hasattr(model, 'module'):
        model = model.module
    img_dir = args.img_dir
    imgs = os.listdir(img_dir)
    for img_name in imgs:
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(img_dir, img_name)
            result = inference_detector(model, img_path)
            out_file = None
            if args.show_dir is not None:
                out_file = os.path.join(args.show_dir, img_name)
            model.show_result(img_path, result, score_thr=args.score_thr, show=True,
                              ood_thr=anomaly_score_threshold,
                              show_bboxes=args.show_bbox,
                              mask_color=(0, 0, 255),
                              out_file=out_file)
##########################################################################



if __name__ == '__main__':
    main()
