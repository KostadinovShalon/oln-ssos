import argparse

import numpy as np
import sklearn.metrics as sk
import json

recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default,
                          pos_label=None, return_index=False):

    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    # import ipdb;
    # ipdb.set_trace()
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]
    recall_fps = fps / fps[-1]
    # breakpoint()
    ## additional code for calculating.
    if return_index:
        recall_level_fps = 1 - recall_level_default
        index_for_tps = threshold_idxs[np.argmin(np.abs(recall - recall_level))]
        index_for_fps = threshold_idxs[np.argmin(np.abs(recall_fps - recall_level_fps))]
        index_for_id_initial = []
        index_for_ood_initial = []
        for index in range(index_for_fps, index_for_tps + 1):
            if y_true[index] == 1:
                index_for_id_initial.append(desc_score_indices[index])
            else:
                index_for_ood_initial.append(desc_score_indices[index])
    # import ipdb;
    # ipdb.set_trace()
    ##
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    # 8.868, ours
    # 5.772, vanilla
    # 5.478, vanilla 18000
    # 6.018, oe
    # 102707,
    # 632
    # 5992
    # breakpoint()
    if return_index:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), index_for_id_initial, index_for_ood_initial
    else:
        return fps[cutoff] / (np.sum(np.logical_not(y_true)))
    # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default, return_index=False, plot=False):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    if plot:
        # breakpoint()
        import matplotlib.pyplot as plt
        fpr1, tpr1, thresholds = sk.roc_curve(labels, examples, pos_label=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr1, tpr1, linewidth=2,
                         label='10000_1')
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.legend(fontsize=12)
        plt.savefig('10000_1.jpg', dpi=250)
    aupr = sk.average_precision_score(labels, examples)
    if return_index:
        fpr, index_id, index_ood = fpr_and_fdr_at_recall(labels, examples, recall_level, return_index=return_index)
        return auroc, aupr, fpr, index_id, index_ood
    else:
        fpr= fpr_and_fdr_at_recall(labels, examples, recall_level)
        return auroc, aupr, fpr


def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector')
parser.add_argument('--id-detections')
parser.add_argument('--ood-detections')
parser.add_argument('--score-threshold', default=1., type=float)
args = parser.parse_args()

score_thr = args.score_threshold
id_detections = json.load(open(args.id_detections, 'r'))
ood_detections = json.load(open(args.ood_detections, 'r'))

id_detections_ood_scores = [a['ood_score'] for a in id_detections if a['score'] > score_thr]
ood_detections_ood_scores = [a['ood_score'] for a in ood_detections if a['score'] > score_thr]
measures = get_measures(id_detections_ood_scores, ood_detections_ood_scores, plot=False)


print_measures(measures[0], measures[1], measures[2], 'energy')
