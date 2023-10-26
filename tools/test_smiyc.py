from argparse import ArgumentParser

import numpy as np
import wandb
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from road_anomaly_benchmark.__main__ import metric
from road_anomaly_benchmark.evaluation import Evaluation

from pycocotools.mask import decode as decode_mask

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--k-pseudo-labels', type=int)
    parser.add_argument('--dataset', default="AnomalyTrack-validation")
    parser.add_argument('--epochs', type=int)
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
    if args.use_wandb:
        wandb.init(project="OLN-VOS",
                   config={
                       "architecture": "oln-vos-mask-deep-pseudo-labels",
                       "dataset": args.dataset,
                       "backbone": "ResNet50",
                       "Learning Rate": 0.001,
                       "epochs": args.epochs,
                       "Pseudo Labels": args.k_pseudo_labels
                   })
    for i in range(args.epochs):
        ckpt = args.checkpoint.rsplit('/', 1)[0] + f"/epoch_{i + 1}.pth"
        model = init_detector(args.config, ckpt, device=args.device)

        # test a single image
        ev = Evaluation(
            method_name=f'OLNVOSMASK_min_k{args.k_pseudo_labels}_deep_epoch_{i + 1}',
            dataset_name=args.dataset,
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
        ag = ev.calculate_metric_from_saved_outputs(
            "PixBinaryClass",
            sample=None,
            parallel=False,
            show_plot=False,
            frame_vis=None,
            default_instancer=True,
        )
        print(f"AP: {ag.area_PRC}, AUROC: {ag.area_ROC}, FPR95: {ag.tpr95_fpr}")
        if args.use_wandb:
            wandb.log({"epoch": i + 1,
                       "ap": ag.area_PRC, "auroc": ag.area_ROC, "fpr95": ag.tpr95_fpr})



if __name__ == '__main__':
    args = parse_args()
    main(args)