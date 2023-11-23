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
import pickle


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


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
    return fts
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
    parser.add_argument('--samples-per-class', type=int, default=5)
    parser.add_argument('--sampling-size', type=int, default=10_000)
    parser
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

    id_coco_file = json.load(open(args.id_annotations, 'r'))
    id_object_features = []
    for img_dict in tqdm.tqdm(id_coco_file['images']):
        fts = extract_features(model, img_dict, id_coco_file, args.id_root_dir, cfg, device)
        if fts is not None:
            id_object_features.append(fts)

    ood_coco_file = json.load(open(args.ood_annotations, 'r'))
    ood_object_features = []
    for img_dict in tqdm.tqdm(ood_coco_file['images']):
        fts = extract_features(model, img_dict, ood_coco_file, args.ood_root_dir, cfg, device)
        if fts is not None:
            ood_object_features.append(fts)

    id_object_features = torch.cat(id_object_features, dim=0)
    ood_object_features = torch.cat(ood_object_features, dim=0)

    kmeans = MiniBatchKMeans(n_clusters=k, n_init=1, batch_size=1024)
    id_labels = kmeans.fit_predict(id_object_features.cpu())
    id_labels = torch.tensor(id_labels).to(device)

    synthetic_fts = []
    for index in range(k):
        kth_class_index = id_labels == index
        kth_id_fts = id_object_features[kth_class_index]
        if index == 0:
            X = kth_id_fts - kth_id_fts.mean(0)
            mean_embed_id = kth_id_fts.mean(0).view(1, -1)
        else:
            X = torch.cat((X, kth_id_fts - kth_id_fts.mean(0)), 0)
            mean_embed_id = torch.cat((mean_embed_id, kth_id_fts.mean(0).view(1, -1)), 0)

        temp_precision = torch.mm(X.t(), X) / len(X)
        # for stable training.
        temp_precision += 0.0001 * torch.eye(1024, device=device)

    for index in range(k):
        new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
            mean_embed_id[index], covariance_matrix=temp_precision)
        for _ in range(args.samples_per_class):
            negative_samples = new_dis.rsample((args.sampling_size,))
            prob_density = new_dis.log_prob(negative_samples)

            # keep the data in the low density area.
            cur_samples, index_prob = torch.topk(- prob_density, 1)
            synthetic_fts.append(negative_samples[index_prob])
            del negative_samples
        del new_dis
    synthetic_fts = torch.cat(synthetic_fts, dim=0)

    total_fts = torch.cat([id_object_features, synthetic_fts, ood_object_features], dim=0)
    tsne = TSNE(n_components=2).fit_transform(total_fts.cpu().numpy())
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    n_id = len(id_object_features)
    n_syn = len(synthetic_fts)
    n_ood = len(ood_object_features)

    plt.figure()
    ax = plt.gca()
    ax.scatter(tx[(n_id + n_syn):], ty[(n_id + n_syn):], color='black', label='test annomalies')
    ax.scatter(tx[:n_id], ty[:n_id], color='blue', label='normal')
    ax.scatter(tx[n_id:(n_id + n_syn)], ty[n_id:(n_id + n_syn)], color='red', label='synthetic outliers')
    ax.legend(loc='best')
    plt.title("Normal vs Outliers features (TSNE n=2)")
    plt.show()

    colors_per_class = [(75, 88, 134), (0, 255, 0), (0, 0, 255), (128, 73, 233), (255, 255, 0),
                        (255, 0, 255), (0, 255, 255), (128, 0, 255), (128, 128, 128), (128, 255, 0),
                        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 0, 128), (128, 128, 0),
                        (255, 128, 0), (0, 255, 128), (255, 0, 128), (128, 128, 255), (255, 255, 128)]
    plt.figure()
    ax = plt.gca()
    ax.scatter(tx[(n_id + n_syn):], ty[(n_id + n_syn):], color='black', marker='v', label='test anomalies')
    for i in range(k):
        # find the samples of the current class in the data
        indices = id_labels.cpu() == i
        indices = torch.cat([indices, torch.tensor([False] * (n_syn + n_ood))], dim=0)

        # extract the coordinates of the points of this class only
        current_tx = tx[indices]
        current_ty = ty[indices]

        # convert the class color to matplotlib format
        color = np.array(colors_per_class[i], dtype=np.float) / 255

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, color=color)
    ax.scatter(tx[n_id:(n_id + n_syn)], ty[n_id:(n_id + n_syn)], color='red', marker='x', label='synthetic outliers')
    ax.legend(loc='best')
    plt.show()

    tsne3 = TSNE(n_components=3).fit_transform(total_fts.cpu().numpy())
    tx3 = tsne3[:, 0]
    ty3 = tsne3[:, 1]
    tz3 = tsne3[:, 2]

    fig = plt.figure(figsize=[16, 8])
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter3D(tx3[(n_id + n_syn):], ty3[(n_id + n_syn):], tz3[(n_id + n_syn):], color='black', label='test anomalies')
    ax.scatter3D(tx3[:n_id], ty3[:n_id], tz3[:n_id], color='blue', label='normal')
    ax.scatter3D(tx3[n_id:(n_id + n_syn)], ty3[n_id:(n_id + n_syn)], tz3[n_id:(n_id + n_syn)], color='red', label='synthetic outliers')
    ax.legend(loc='best')
    plt.title("Normal vs Outliers features (TSNE n=3)")
    pickle.dump(fig, open('tsne3_outliers.pickle', 'wb'))
    plt.show()

    fig = plt.figure(figsize=[16, 8])
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter3D(tx3[(n_id + n_syn):], ty3[(n_id + n_syn):], tz3[(n_id + n_syn):], color='black', marker='v', label='test anomalies')
    for i in range(k):
        # find the samples of the current class in the data
        indices = id_labels.cpu() == i
        indices = torch.cat([indices, torch.tensor([False] * (n_syn + n_ood))], dim=0)

        # extract the coordinates of the points of this class only
        current_tx = tx3[indices]
        current_ty = ty3[indices]
        current_tz = tz3[indices]

        # convert the class color to matplotlib format
        color = np.array(colors_per_class[i], dtype=np.float) / 255

        # add a scatter plot with the corresponding color and label
        ax.scatter3D(current_tx, current_ty, current_tz, color=color)
    ax.scatter3D(tx3[n_id:(n_id + n_syn)], ty3[n_id:(n_id + n_syn)], tz3[n_id:(n_id + n_syn)], color='red', marker='x', label='synthetic outliers')
    ax.legend(loc='best')
    pickle.dump(fig, open('tsne3_outliers_per_class.pickle', 'wb'))
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
