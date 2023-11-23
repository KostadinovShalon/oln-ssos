import json
import os.path
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import scipy.stats as st

from mmdet.apis import init_detector
from mmdet.core import bbox2roi
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def extract_features(model, img_dict, coco_file, root_dir, cfg, device):
    # add information into dict
    img = os.path.join(root_dir, img_dict['file_name'])
    anns = [a for a in coco_file['annotations'] if a['image_id'] == img_dict['id']]
    if len(anns) == 0:
        return
    proposals = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                  a['bbox'][1] + a['bbox'][3], 1] for a in anns]
    proposals = torch.tensor(proposals).to(device)
    data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    proposals[:, :4] = proposals[:, :4] * torch.tensor(data['img_metas'][0][0]['scale_factor']).to(device)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        x = model.extract_feat(img=data['img'][0])
        rois = bbox2roi([proposals])
        bbox_results = model.roi_head._bbox_forward(x, rois)
        fts = bbox_results['shared_bbox_feats']
        pseudo_scores = model.roi_head.pseudo_score(fts)
        energies = torch.logsumexp(pseudo_scores, 1)
        ood_scores = model.roi_head.logistic_regression_layer(energies.view(-1, 1)).sigmoid()
    return fts, pseudo_scores.argmax(dim=1), ood_scores
    # object_features.append(fts)
    # labels.append(pseudo_scores.argmax(dim=1))
    # scores.append(ood_scores)
    # labels.append(torch.tensor([a['category_id'] for a in anns]))
    # result = model(img=data['img'], img_metas=data['img_metas'],
    #                return_loss=False, proposals=[[proposals]])[0]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--id-annotations', help='Annotations file')
    parser.add_argument('--id-root-dir', help='id images root dir')
    parser.add_argument('--ood-annotations', help='OoD Annotations file')
    parser.add_argument('--ood-root-dir', help='ood images root dir')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    device = next(model.parameters()).device  # model device
    cfg = model.cfg
    # prepare data

    k = args.k

    ood_coco_file = json.load(open(args.ood_annotations, 'r'))
    ood_object_features = []
    ood_scores = []
    ood_labels = []
    for img_dict in tqdm.tqdm(ood_coco_file['images']):
        data = extract_features(model, img_dict, ood_coco_file, args.ood_root_dir, cfg, device)
        if data is None:
            continue
        fts, label, ood_score = data
        ood_object_features.append(fts)
        ood_scores.append(ood_score)
        ood_labels.append(label)

    id_coco_file = json.load(open(args.id_annotations, 'r'))
    id_object_features = []
    id_scores = []
    id_labels = []
    for img_dict in tqdm.tqdm(id_coco_file['images']):
        data = extract_features(model, img_dict, id_coco_file, args.id_root_dir, cfg, device)
        if data is None:
            continue
        fts, label, ood_score = data
        id_object_features.append(fts)
        id_scores.append(ood_score)
        id_labels.append(label)

    id_object_features = torch.cat(id_object_features, dim=0).cpu().numpy()
    id_scores = torch.cat(id_scores, dim=0).cpu().flatten().numpy()
    id_labels = torch.cat(id_labels, dim=0).cpu().numpy()

    ood_object_features = torch.cat(ood_object_features, dim=0).cpu().numpy()
    ood_scores = torch.cat(ood_scores, dim=0).cpu().flatten().numpy()
    ood_labels = torch.cat(ood_labels, dim=0).cpu().numpy()
    min_id_score = id_scores.min()
    min_ood_score = ood_scores.min()
    min_val = min(min_id_score, min_ood_score)
    kde_xs = np.linspace(min_val, 1, 100)
    id_kde = st.gaussian_kde(id_scores)
    ood_kde = st.gaussian_kde(ood_scores)

    plt.plot(kde_xs, id_kde.pdf(kde_xs), label='Normal', color='blue')
    plt.plot(kde_xs, ood_kde.pdf(kde_xs), label='Anomaly', color='red')
    plt.fill_between(kde_xs, 0, id_kde.pdf(kde_xs), color='blue', alpha=.25)
    plt.fill_between(kde_xs, 0, ood_kde.pdf(kde_xs), color='red', alpha=.25)
    plt.legend(loc="upper right")
    plt.title("Anomaly scores")
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
