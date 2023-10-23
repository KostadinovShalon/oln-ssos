import argparse
import os
import warnings

import mmcv
import numpy as np
import sklearn.metrics
import torch
import tqdm
from mmcv import DictAction, Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, wrap_fp16_model, load_checkpoint, get_dist_info

from mmdet.apis import single_gpu_test, multi_gpu_test
from mmdet.datasets import replace_ImageToTensor, build_dataset, build_dataloader
from mmdet.models import build_detector
from pycocotools.mask import decode as decode_mask

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path for the in-distribution dataset')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format of the id dataset')
    parser.add_argument('--show', action='store_true', help='show id results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def get_ap_fpr95(gt_mask, ood_pred):
    valid_gt_mask = gt_mask[gt_mask < 255]
    valid_ood_pred = ood_pred[gt_mask < 255]
    auprc = sklearn.metrics.average_precision_score(valid_gt_mask, valid_ood_pred)

    # Getting FPR95
    anomaly_gt_indices = valid_gt_mask == 1
    anomaly_ood_scores = valid_ood_pred[anomaly_gt_indices]
    sorted_ood_scores = np.sort(anomaly_ood_scores)
    # This threshold indicates that everything above it will be detected as anomaly
    # Since we want to get 95% of the anomalies, we do this:
    tpr95_threshold = sorted_ood_scores[int(0.05 * len(sorted_ood_scores))]

    normal_gt_pixels = sum(valid_gt_mask == 0)
    # Now we get, how many of those pixels that should be 0 are marked above the threshold
    fp_pixels = sum(valid_ood_pred[valid_gt_mask == 0] > tpr95_threshold)

    # FPR is considered as how many false anomalies I detected with respect to the normal instances
    fpr95 = fp_pixels / normal_gt_pixels
    return auprc, fpr95


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg_options = args.cfg_options
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        show = args.show
        show_dir = args.show_dir
        sc_th = args.show_score_thr
        outputs = single_gpu_test(model, data_loader, show, show_dir,
                                  sc_th)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    avg_auprc = avg_fpr95 = 0
    min_auprc = min_fpr95 = 0
    max_auprc = max_fpr95 = 0
    for i, x in enumerate(tqdm.tqdm(outputs)):
        data_root = 'data/roadanomaly21/'
        filename = dataset.data_infos[i]['filename']
        filename = filename[:-4] + "_labels_semantic.png"
        label_img_path = os.path.join(data_root, 'labels_masks', filename)
        gt_mask = cv2.imread(label_img_path)[:, :, 0]

        masks = [decode_mask(m) for m in x[1][0]]

        segm_maps = np.stack(masks, axis=0)
        ood_scores = x[0][0][:, 5]
        count = np.sum(segm_maps, axis=0)
        ood = ood_scores[:, None, None] * segm_maps
        sum_ood = np.sum(ood, axis=0)
        max_ood = np.max(ood, axis=0)
        avg = sum_ood / count
        avg[np.isnan(avg)] = 1.0
        max_ood[max_ood == 0] = 1.
        ood[ood == 0] = 1
        min_ood = np.min(ood, axis=0)

        avg = 1 - avg
        min_ood = 1 - min_ood
        max_ood = 1 - max_ood
        _avg_auprc, _avg_fpr95 = get_ap_fpr95(gt_mask, avg)
        avg_auprc += _avg_auprc
        avg_fpr95 += _avg_fpr95

        _min_auprc, _min_fpr95 = get_ap_fpr95(gt_mask, min_ood)
        min_auprc += _min_auprc
        min_fpr95 += _min_fpr95

        _max_auprc, _max_fpr95 = get_ap_fpr95(gt_mask, max_ood)
        max_auprc += _max_auprc
        max_fpr95 += _max_fpr95
    print(f"AuPRC and FPR95 for avg method: {avg_auprc / len(outputs)}, {avg_fpr95 / len(outputs)}")
    print(f"AuPRC and FPR95 for min method: {min_auprc / len(outputs)}, {min_fpr95 / len(outputs)}")
    print(f"AuPRC and FPR95 for max method: {max_auprc / len(outputs)}, {max_fpr95 / len(outputs)}")


if __name__ == '__main__':
    main()
