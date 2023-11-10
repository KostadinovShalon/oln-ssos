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

from mmdet.apis import init_detector
from mmdet.core import bbox2roi
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('annotations', help='Annotations file')
    parser.add_argument('root_dir', help='Annotations file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    device = next(model.parameters()).device  # model device
    cfg = model.cfg
    # prepare data

    coco_file = json.load(open(args.annotations, 'r'))
    object_features = []
    labels = []
    images = []
    for img_dict in tqdm.tqdm(coco_file['images']):
        # add information into dict
        img = os.path.join(args.root_dir, img_dict['file_name'])
        anns = [a for a in coco_file['annotations'] if a['image_id'] == img_dict['id']]
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
        norm_cfg = data['img_metas'][0][0]['img_norm_cfg']
        # Cropping images
        for p in proposals:
            x1, y1, x2, y2 = p[:4]
            cropped_img = data['img'][0][0, :, int(y1):int(y2), int(x1):int(x2)]
            cropped_img = cropped_img.permute((1, 2, 0)).cpu().numpy()
            cropped_img = (cropped_img * norm_cfg['std']) + norm_cfg['mean']
            images.append(cropped_img)

        # forward the model
        with torch.no_grad():
            x = model.extract_feat(img=data['img'][0])
            rois = bbox2roi([proposals])
            bbox_results = model.roi_head._bbox_forward(x, rois)
            fts = bbox_results['shared_bbox_feats']
            pseudo_scores = model.roi_head.pseudo_score(fts)
            energies = torch.logsumexp(pseudo_scores, 1)
            ood_scores = model.roi_head.logistic_regression_layer(energies.view(-1, 1)).sigmoid()
            object_features.append(fts)
            labels.append(pseudo_scores.argmax(dim=1))
            # labels.append(torch.tensor([a['category_id'] for a in anns]))
            # result = model(img=data['img'], img_metas=data['img_metas'],
            #                return_loss=False, proposals=[[proposals]])[0]
    object_features = torch.cat(object_features, dim=0).cpu().numpy()

    # kmeans = MiniBatchKMeans(n_clusters=5, n_init=1, batch_size=1000).fit(object_features)
    # labels = kmeans.predict(object_features)
    labels = torch.cat(labels, dim=0).cpu().numpy()
    tsne = TSNE(n_components=2).fit_transform(object_features)

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    categories = {1: 'firearms', 2: 'firearm parts'}
    # categories = {3: 'knife', 4: 'camera', 5: 'ceramic knife', 6: 'laptop'}
    categories = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    colors_per_class = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 0)]

    for i, (label, name) in enumerate(categories.items()):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = np.array(colors_per_class[i], dtype=np.float) / 255

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=name)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()

    # Compute the coordinates of the image on the plot
    def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
        image_height, image_width, _ = image.shape

        # compute the image center coordinates on the plot
        center_x = int(image_centers_area_size * x) + offset

        # in matplotlib, the y axis is directed upward
        # to have the same here, we need to mirror the y coordinate
        center_y = int(image_centers_area_size * (1 - y)) + offset

        # knowing the image center,
        # compute the coordinates of the top left and bottom right corner
        tl_x = center_x - int(image_width / 2)
        tl_y = center_y - int(image_height / 2)

        br_x = tl_x + image_width
        br_y = tl_y + image_height

        return tl_x, tl_y, br_x, br_y

    plot_size = 1100
    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
    max_image_size = 50

    def scale_image(img, size):
        h, w = img.shape[:2]
        hf, wf = h / size, w / size
        f = max(hf, wf)
        new_size = (int(w / f), int(h / f))
        return cv2.resize(img, new_size)

    def draw_rectangle_by_class(img, color):
        img[:2, :] = color
        img[-2:, :] = color
        img[:, -2:] = color
        img[:, :2] = color
        return img

    for image, label, x, y in tqdm.tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, colors_per_class[label])

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, 1000, 50)

        # put the image to its t-SNE coordinates using numpy sub-array indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image
    fig = plt.figure(figsize=(20, 20))
    fig.patch.set_visible(False)
    ax = plt.gca()
    ax.axis('off')
    plt.imshow(tsne_plot)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)