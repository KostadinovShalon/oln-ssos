import json
import os
from argparse import ArgumentParser

import cv2
import torch
from matplotlib import pyplot as plt
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np

from mmdet.apis import init_detector
from mmdet.core import bbox2roi
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
import random

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--annotations', help='Annotations file')
    parser.add_argument('--root-dir', help='images root dir')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--sampling-size', type=int, default=100)
    parser.add_argument('--use-k', type=int, default=0)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    args = parser.parse_args()
    return args


def get_image_and_annotations(img_dict, anns_dict, root_dir, return_cropped_objects=False):
    """
    Get image and annotations from coco file
    """
    img = os.path.join(root_dir, img_dict['file_name'])
    anns = [a for a in anns_dict if a['image_id'] == img_dict['id']]
    cropped_objects = None
    _img = cv2.imread(img)
    if _img is None:
        return None, None, None
    if return_cropped_objects:
        cropped_objects = []
        for a in anns:
            cropped_objects.append(_img[int(a['bbox'][1]):int(a['bbox'][1] + a['bbox'][3]),
                                        int(a['bbox'][0]):int(a['bbox'][0] + a['bbox'][2])])
    return img, anns, cropped_objects


def extract_features(model, img, anns, cfg, device):
    """
    Extract object features given an image 'img' and the ground truth annotations 'anns'

    It returns the features of the objects in the image in an N x C tensor
    """
    # add information into dict
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
    if len(proposals) == 0:
        return
    proposals[:, :4] = proposals[:, :4] * torch.tensor(data['img_metas'][0][0]['scale_factor']).to(device)
    proposals = proposals.to(torch.float32)
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
        bbox_feats = model.roi_head.bbox_roi_extractor(
            x[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
        if model.roi_head.with_shared_head:
            bbox_feats = model.roi_head.shared_head(bbox_feats)
        _, _, _, fts = model.roi_head.bbox_head(bbox_feats)
    return fts


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
    device = args.device

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    device = next(model.parameters()).device  # model device
    cfg = model.cfg
    # prepare data
    coco_file = json.load(open(args.annotations, 'r'))
    object_features = []

    cropped_images = []
    anns_dict = coco_file['annotations']

    # Shuffle coco_file['images']
    random.shuffle(coco_file['images'])

    k = model.roi_head.k

    if args.use_k <= 0:  # Use all pseudo classes
        for img_dict in tqdm(coco_file['images'][:args.sampling_size]):
            img, anns, _cropped_images = get_image_and_annotations(img_dict, anns_dict, args.root_dir, return_cropped_objects=True)
            if img is None:
                continue
            fts = extract_features(model, img, anns, cfg, device)
            if fts is not None:
                object_features.append(fts)
            if _cropped_images is not None:
                cropped_images.extend(_cropped_images)
        object_features = torch.cat(object_features, dim=0)
        pseudo_label_scores = model.roi_head.pseudo_score(object_features)  # N x C logits
        pseudo_labels = pseudo_label_scores.argmax(dim=1)
        pseudo_labels = pseudo_labels.cpu().numpy()
    else:
        # Use only args.use_k pseudo classes
        k_to_use = random.sample(range(k), args.use_k)
        pseudo_labels = []
        i = 0

        # Sample images until we have args.sampling_size samples
        while len(cropped_images) < args.sampling_size:
            img_dict = coco_file['images'][i]
            img, anns, _cropped_images = get_image_and_annotations(img_dict, anns_dict, args.root_dir, return_cropped_objects=True)
            i += 1
            if img is None:
                continue
            fts = extract_features(model, img, anns, cfg, device)
            if fts is None:
                continue
            pseudo_label_scores = model.roi_head.pseudo_score(fts)
            _labels = pseudo_label_scores.argmax(dim=1).cpu().tolist()
            for cropped_img, p, _fts in zip(_cropped_images, _labels, fts):
                if p in k_to_use:
                    cropped_images.append(cropped_img)
                    pseudo_labels.append(p)
                    object_features.append(_fts)
        pseudo_labels = np.array(pseudo_labels)
        object_features = torch.stack(object_features, dim=0)

    # At this point, object features is a tensor of shape N x C, where N is the number of objects and C is the number of features
    # and pseudo_labels is a numpy array of shape N containing the pseudo labels of each object

    # tSNE plot
    tsne = TSNE(n_components=2).fit_transform(object_features.cpu().numpy())
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    plt.figure()
    ax = plt.gca()

    # Define colors and markers for each pseudo class
    colors_per_class = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 73, 233), (255, 255, 0),
                        (255, 0, 255), (0, 255, 255), (128, 0, 255), (128, 128, 128), (128, 255, 0),
                        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 0, 128), (128, 128, 0),
                        (255, 128, 0), (0, 255, 128), (255, 0, 128), (128, 128, 255), (255, 255, 128)]
    markers = ['o', '^', 'v', 's', 'x', "D", '+', "<", ">", "P"]

    # Iterate over each pseudo class or only the args.use_k pseudo classes
    list_to_iterate = range(k) if args.use_k <= 0 else k_to_use
    for index in list_to_iterate:
        # Get the indexes of the objects that belong to the current pseudo class
        kth_class_index = pseudo_labels == index

        # Check if the list will be empty
        if not np.any(kth_class_index):
            continue
        # Get the color for the current pseudo class
        color = np.array(colors_per_class[index % len(colors_per_class)], dtype=np.float) / 255
        # Scatter plot the objects in the current pseudo class
        ax.scatter(tx[kth_class_index], ty[kth_class_index],
                   color=color,
                   marker=markers[index // len(colors_per_class)],
                   label=f'pc={index}')
    ax.legend(loc='best')
    plt.title("Pseudoclass distribution (TSNE n=2)")
    plt.show()

    # tSNE plot with cropped image patches
    plot_size = 512
    max_patch_size = 64
    min_patch_size = 16
    tsne_plot = 255 * np.ones((plot_size + max_patch_size * 2, plot_size + max_patch_size * 2, 3), np.uint8)

    # Iterateve over each object, and its pseudo label, and plot the cropped image patch
    for cropped_image, pseudo_label, _x, _y in zip(cropped_images, pseudo_labels, tx, ty):
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        color = np.array(colors_per_class[pseudo_label % len(colors_per_class)], dtype=np.float)
        # Scale cropped image for plotting keeping aspect ratio
        if cropped_image.shape[0] < min_patch_size or cropped_image.shape[1] < min_patch_size:
            if cropped_image.shape[0] < cropped_image.shape[1]:
                scale = min_patch_size / cropped_image.shape[0]
                cropped_image = cv2.resize(cropped_image, (int(cropped_image.shape[1] * scale), min_patch_size))
            else:
                scale = min_patch_size / cropped_image.shape[1]
                cropped_image = cv2.resize(cropped_image, (min_patch_size, int(cropped_image.shape[0] * scale)))
        if cropped_image.shape[0] > max_patch_size or cropped_image.shape[1] > max_patch_size:
            if cropped_image.shape[0] > cropped_image.shape[1]:
                scale = max_patch_size / cropped_image.shape[0]
                cropped_image = cv2.resize(cropped_image, (int(cropped_image.shape[1] * scale), max_patch_size))
            else:
                scale = max_patch_size / cropped_image.shape[1]
                cropped_image = cv2.resize(cropped_image, (max_patch_size, int(cropped_image.shape[0] * scale)))
        _cropped_shape = cropped_image.shape
        # Draw rectangular border
        img = cv2.rectangle(cropped_image, (0, 0), (_cropped_shape[1], _cropped_shape[0]), color, 3)
        top_left_corner = (int(_x * plot_size), int((1 - _y) * plot_size))
        tsne_plot[top_left_corner[1]:top_left_corner[1] + _cropped_shape[0],
                  top_left_corner[0]:top_left_corner[0] + _cropped_shape[1]] = img
    plt.imshow(tsne_plot)
    plt.axis('off')
    plt.show()

    # tsne3 = TSNE(n_components=3).fit_transform(object_features.cpu().numpy())  # fts: N x C, Cluster: N
    # tx3 = tsne3[:, 0]
    # ty3 = tsne3[:, 1]
    # tz3 = tsne3[:, 2]
    #
    # fig = plt.figure(figsize=[16, 8])
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # for index in range(k):
    #     color = np.array(colors_per_class[index % len(colors_per_class)], dtype=np.float) / 255
    #     kth_class_index = pseudo_labels == index
    #     ax.scatter3D(tx3[kth_class_index][:20], ty3[kth_class_index][:20], tz3[kth_class_index][:20],
    #                color=color,
    #                marker=markers[index // len(colors_per_class)],
    #                label=f'pc={index}')
    # ax.legend(loc='best')
    # plt.title("Pseudoclass distribution (TSNE n=3)")
    # # pickle.dump(fig, open('tsne3_outliers.pickle', 'wb'))
    # plt.show()


if __name__ == '__main__':
    main()
