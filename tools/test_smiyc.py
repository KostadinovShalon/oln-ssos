from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from road_anomaly_benchmark.evaluation import Evaluation

from pycocotools.mask import decode as decode_mask

def parse_args():
    parser = ArgumentParser()
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
    ev = Evaluation(
        method_name='OLNVOSMASK_min_k5',
        dataset_name='AnomalyTrack-validation',
        # dataset_name = 'AnomalyTrack-test',
    )
    for frame in tqdm(ev.get_frames()):
        # run method here
        result = inference_detector(model, frame.image)
        masks = [m for m in result[1][0]]

        segm_maps = np.stack(masks, axis=0)
        ood_scores = result[0][0][:, 5]
        # count = np.sum(segm_maps, axis=0)
        ood = ood_scores[:, None, None] * segm_maps
        # sum_ood = np.sum(ood, axis=0)
        # max_ood = np.max(ood, axis=0)
        # avg = sum_ood / count
        # avg[np.isnan(avg)] = 1.0
        # max_ood[max_ood == 0] = 1.
        ood[ood == 0] = 1
        min_ood = np.min(ood, axis=0)

        # avg = 1 - avg
        min_ood = 1 - min_ood
        # max_ood = 1 - max_ood
        # provide the output for saving
        ev.save_output(frame, min_ood)
    ev.wait_to_finish_saving()
    # show the results


if __name__ == '__main__':
    args = parse_args()
    main(args)