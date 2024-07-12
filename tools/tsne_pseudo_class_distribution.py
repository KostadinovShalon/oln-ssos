import json
import os
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
from wandb.util import np

from mmdet.apis import init_detector
from mmdet.core import bbox2roi
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--annotations', help='Annotations file')
    parser.add_argument('--root-dir', help='images root dir')
    parser.add_argument('--k', type=int, default=80)
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--sampling-size', type=int, default=100)
    parser.add_argument('--use-temp-fts', action="store_true")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    args = parser.parse_args()
    return args


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
        bbox_feats = model.roi_head.pseudo_bbox_roi_extractor(
            x[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = bbox_feats.flatten(1)
    return bbox_feats


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def main():
    args = parse_args()
    k = args.k
    device = args.device
    if not args.use_temp_fts:
        # build the model from a config file and a checkpoint file
        model = init_detector(args.config, args.checkpoint, device=args.device)
        # test a single image
        device = next(model.parameters()).device  # model device
        cfg = model.cfg
        # prepare data
        coco_file = json.load(open(args.annotations, 'r'))
        object_features = []
        for img_dict in tqdm(coco_file['images']):
            fts = extract_features(model, img_dict, coco_file, args.root_dir, cfg, device)
            if fts is not None:
                object_features.append(fts)

        object_features = torch.cat(object_features, dim=0)
        np.save(open("temp_fts.npy", "wb"), object_features.cpu().numpy())
    else:
        object_features = torch.tensor(np.load(open("temp_fts.npy", "rb")))

    kmeans = MiniBatchKMeans(n_clusters=k, n_init=1, batch_size=1024)
    labels = kmeans.fit_predict(object_features.cpu())
    # labels = torch.tensor(labels).to(device)

    tsne = TSNE(n_components=2).fit_transform(object_features.cpu().numpy())
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    plt.figure()
    ax = plt.gca()

    colors_per_class = [(75, 88, 134), (0, 255, 0), (0, 0, 255), (128, 73, 233), (255, 255, 0),
                        (255, 0, 255), (0, 255, 255), (128, 0, 255), (128, 128, 128), (128, 255, 0),
                        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 0, 128), (128, 128, 0),
                        (255, 128, 0), (0, 255, 128), (255, 0, 128), (128, 128, 255), (255, 255, 128)]
    markers = ['o', '^', 'v', 's', 'x', "D", '+', "<", ">", "P"]

    for index in range(k):
        kth_class_index = labels == index
        color = np.array(colors_per_class[index % len(colors_per_class)], dtype=np.float) / 255
        ax.scatter(tx[kth_class_index][:20], ty[kth_class_index][:20],
                   color=color,
                   marker=markers[index // len(colors_per_class)],
                   label=f'pc={index}')
    ax.legend(loc='best')
    plt.title("Pseudoclass distribution (TSNE n=2)")
    plt.show()

    tsne3 = TSNE(n_components=3).fit_transform(object_features.cpu().numpy())  # fts: N x C, Cluster: N
    tx3 = tsne3[:, 0]
    ty3 = tsne3[:, 1]
    tz3 = tsne3[:, 2]

    fig = plt.figure(figsize=[16, 8])
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for index in range(k):
        color = np.array(colors_per_class[index % len(colors_per_class)], dtype=np.float) / 255
        kth_class_index = labels == index
        ax.scatter3D(tx3[kth_class_index][:20], ty3[kth_class_index][:20], tz3[kth_class_index][:20],
                   color=color,
                   marker=markers[index // len(colors_per_class)],
                   label=f'pc={index}')
    ax.legend(loc='best')
    plt.title("Pseudoclass distribution (TSNE n=3)")
    # pickle.dump(fig, open('tsne3_outliers.pickle', 'wb'))
    plt.show()


if __name__ == '__main__':
    main()
