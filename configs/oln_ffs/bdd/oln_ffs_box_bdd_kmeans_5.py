_base_ = ['../../oln_box/oln_box_model.py',
          '../../_base_/datasets/bdd_detection.py',
          '../../_base_/schedules/schedule_1x.py',
          '../../_base_/default_runtime.py'
          ]

custom_imports = dict(
    imports=[
        'vos.models.detectors.epoch_faster_rcnn',
        'vos.models.roi_heads.oln_ffs_roi_head',
        'vos.models.roi_heads.bbox_heads.oln_vos_bbox_head',
        'vos.datasets.vos_coco',
        'vos.datasets.pipelines.loading',
        'vos.datasets.pipelines.formating'
    ],
    allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsWithAnnID', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PseudoLabelFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_ann_ids', 'gt_pseudo_labels',
                               'gt_weak_bboxes', 'gt_weak_bboxes_labels']),
]

model = dict(
    type='EpochFasterRCNN',
    calculate_pseudo_labels_from_epoch=0,
    roi_head=dict(
        type='OLNKMeansFFSRoIHead',
        start_epoch=4,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=100,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        pseudo_label_loss_weight=1.,
        nll_loss_weight=0.0001,
        k=5,
        repeat_ood_sampling=4,
        use_all_proposals_ood=False,
        bbox_head=dict(
            type='VOSShared2FCBBoxScoreHead',
            reg_class_agnostic=True))
)

checkpoint_config = dict(interval=1)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[4])
total_epochs = 8

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
            pipeline=train_pipeline),
)

load_from = './work_dirs/oln_box_bdd_bs2/epoch_8.pth'
custom_hooks = [dict(type='SetEpochInfoHook')]
