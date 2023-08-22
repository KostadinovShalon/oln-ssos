import argparse
import json

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vos.utils.metrics import get_measures, print_measures


def parse_args():
    parser = argparse.ArgumentParser(
        description='OOD metrics evaluator')
    parser.add_argument('id_gt', help='In-distribution ground truth coco json file')
    parser.add_argument('id_results', help='In-distribution results json file')
    parser.add_argument('ood_results', help='Out-of-distribution results json file')
    parser.add_argument('--optimal-score', type=float, default=-1)
    parser.add_argument('--tpr-thres', help='TPR in-distribution rate for threshold calculation',
                        type=float, default=0.95)
    args = parser.parse_args()
    return args


def main(args):
    id_results = json.load(open(args.id_results, 'rb'))
    ood_results = json.load(open(args.ood_results, 'rb'))

    optimal_score_threshold = args.optimal_score

    if optimal_score_threshold < 0:

        gt_coco_api = COCO(args.id_gt)
        res_coco_api = gt_coco_api.loadRes(id_results)
        results_api = COCOeval(gt_coco_api, res_coco_api, iouType='bbox')

        results_api.params.catIds = np.array(range(1, 21))  # TODO: Change to general cases

        # Calculate and print aggregate results
        results_api.evaluate()
        results_api.accumulate()
        results_api.summarize()

        # Compute optimal micro F1 score threshold. We compute the f1 score for
        # every class and score threshold. We then compute the score threshold that
        # maximizes the F-1 score of every class. The final score threshold is the average
        # over all classes.
        precisions = results_api.eval['precision'].mean(0)[:, :, 0, 2]
        recalls = np.expand_dims(results_api.params.recThrs, 1)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        optimal_f1_score = f1_scores.argmax(0)
        scores = results_api.eval['scores'].mean(0)[:, :, 0, 2]
        optimal_score_threshold = [scores[optimal_f1_score_i, i]
                                   for i, optimal_f1_score_i in enumerate(optimal_f1_score)]
        optimal_score_threshold = np.array(optimal_score_threshold)
        optimal_score_threshold = optimal_score_threshold[optimal_score_threshold != 0]
        optimal_score_threshold = optimal_score_threshold.mean()
        print("Optimal score threshold: ", optimal_score_threshold)

    id_inter_feats = [torch.tensor(id_r['inter_feats']) for id_r in id_results if id_r['score'] > optimal_score_threshold]
    id_inter_feats = torch.logsumexp(torch.stack(id_inter_feats)[:, :-1], dim=1).sigmoid().cpu().data.numpy()
    id_scores = [id_r['ood_score'] for id_r in id_results if id_r['score'] > optimal_score_threshold]

    ood_inter_feats = [torch.tensor(ood_r['inter_feats']) for ood_r in ood_results if ood_r['score'] > optimal_score_threshold]
    ood_inter_feats = torch.logsumexp(torch.stack(ood_inter_feats)[:, :-1], dim=1).sigmoid().cpu().data.numpy()
    ood_scores = [ood_r['ood_score'] for ood_r in ood_results if ood_r['score'] > optimal_score_threshold]

    id_scores.sort()
    t = int(len(id_scores) * (1 - args.tpr_thres))-1
    sc_th = id_scores[t]

    print(len(id_scores))
    print(len(ood_scores))
    print(f"Mean OOD score for the ID dataset: {sum(id_scores) / len(id_scores)}")
    print(f"Mean OOD score for the OOD dataset: {sum(ood_scores) / len(ood_scores)}")
    print(f"Median OOD score for the ID dataset: {id_scores[len(id_scores)//2]}")
    print(f"Median OOD score for the OOD dataset: {ood_scores[len(ood_scores)//2]}")

    tpr = sum([1 for s in id_scores if s > sc_th]) / len(id_scores)
    fpr = sum([1 for s in ood_scores if s > sc_th]) / len(ood_scores)

    print(f"TPR in the ID dataset: {tpr}")
    print(f"FPR in the OOD dataset: {fpr}")

    id_inter_feats.sort()
    t = int(len(id_inter_feats) * (1 - args.tpr_thres)) - 1
    sc_th = id_inter_feats[t]

    tpr = len(id_inter_feats[id_inter_feats > sc_th]) / len(id_inter_feats)
    fpr = len(ood_inter_feats[ood_inter_feats > sc_th]) / len(ood_inter_feats)

    print(f"TPR in the ID dataset, v2: {tpr}")
    print(f"FPR in the OOD dataset, v2: {fpr}")

    measures = get_measures(id_inter_feats, ood_inter_feats)
    print_measures(*measures, method_name='BBB')


if __name__ == '__main__':
    args = parse_args()
    main(args)
