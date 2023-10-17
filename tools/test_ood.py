import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('id_config', help='test config file path for the in-distribution dataset')
    parser.add_argument('ood_config', help='test config file path for the out-of-distribution dataset')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-id', help='output result file in pickle format of the id dataset')
    parser.add_argument('--out-ood', help='output result file in pickle format of the ood dataset')
    parser.add_argument('--optimal-score-threshold', type=float)
    parser.add_argument('--anomaly-threshold', type=float)
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval-ood',
        action='store_true',
        help='Eval OOD')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC. This is only done for the id dataset')
    parser.add_argument('--show-id', action='store_true', help='show id results')
    parser.add_argument('--show-ood', action='store_true', help='show ood results')
    parser.add_argument(
        '--show-id-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-ood-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--id-cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--ood-cfg-options',
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


def main():
    args = parse_args()

    assert args.out_id or args.eval or args.format_only or args.show_id or args.show_ood \
        or args.show_id_dir or args.show_ood_dir or args.out_ood or args.eval_ood, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out-id", "--out-ood", "--eval"', "--eval-ood",
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if (args.out_id is not None and not args.out_id.endswith(('.pkl', '.pickle'))) or (
            args.out_ood is not None and not args.out_ood.endswith(('.pkl', '.pickle'))
    ):
        raise ValueError('The output file must be a pkl file.')

    id_cfg = Config.fromfile(args.id_config)
    ood_cfg = Config.fromfile(args.ood_config)
    optimal_score_threshold = args.optimal_score_threshold
    anomaly_score_threshold = args.anomaly_threshold
    for cfg, cfg_options in [(id_cfg, args.id_cfg_options), (ood_cfg, args.ood_cfg_options)]:
        if cfg_options is not None:
            id_cfg.merge_from_dict(cfg_options)
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
        init_dist(args.launcher, **id_cfg.dist_params)

    # build the dataloader
    results = dict(id=list(), ood=list())
    datasets = dict(id=None, ood=None)
    for cfg, cfg_label in [(id_cfg, 'id'), (ood_cfg, 'ood')]:
        if cfg_label == 'id' and optimal_score_threshold is not None and anomaly_score_threshold is not None:
            print('Jumping directly to OOD ')
            continue

        datasets[cfg_label] = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            datasets[cfg_label],
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        model.anomaly_score_threshold = anomaly_score_threshold if anomaly_score_threshold is not None else 1.
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = datasets[cfg_label].CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            show = args.show_id if cfg_label == 'id' else args.show_ood
            show_dir = args.show_id_dir if cfg_label == 'id' else args.show_ood_dir
            sc_th = args.show_score_thr if cfg_label == 'id' else optimal_score_threshold
            outputs = single_gpu_test(model, data_loader, show, show_dir,
                                      sc_th)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
        results[cfg_label] = outputs

        rank, _ = get_dist_info()
        if rank == 0:
            out_file = args.out_id if cfg_label == 'id' else args.out_ood
            if out_file:
                print(f'\nwriting {cfg_label} results to {out_file}')
                mmcv.dump(results[out_file], out_file)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                datasets[cfg_label].format_results(results[cfg_label], **kwargs)
            if args.eval:
                eval_kwargs = id_cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                print(datasets[cfg_label].evaluate(results[cfg_label], **eval_kwargs))
            if args.eval_ood and cfg_label == 'id':
                if optimal_score_threshold is None:
                    print(f"\ngetting id optimal score threshold")
                    if isinstance(results['id'][0], list):
                        id_json_results = datasets['id']._det2json(results['id'])
                        # ood_json_results = datasets['ood']._det2json(results['ood'])
                    elif isinstance(results['id'][0], tuple):
                        _, id_json_results = datasets['id']._segm2json(results['id'])
                        # ood_json_results = datasets['ood']._segm2json(results['ood'])
                    gt_coco_api = COCO(id_cfg.data.test.ann_file)
                    res_coco_api = gt_coco_api.loadRes(id_json_results)
                    results_api = COCOeval(gt_coco_api, res_coco_api, iouType='bbox')

                    results_api.params.catIds = np.array([1])

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
                if anomaly_score_threshold is None:
                    optimal_results = [r for r in id_json_results if r['score'] > optimal_score_threshold]
                    ood_scores = [o['ood_score'] for o in optimal_results]
                    ood_scores.sort()
                    anomaly_score_threshold = ood_scores[int(len(ood_scores) * 0.05)]
                print("Anomaly Score Threshold: ", anomaly_score_threshold)


if __name__ == '__main__':
    main()
