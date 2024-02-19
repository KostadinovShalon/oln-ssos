_base_ = [
    '../../oln_box/oln_box.py',
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
# model settings
model = dict(
    type='EpochFasterRCNN',
    calculate_pseudo_labels_from_epoch=0,
    roi_head=dict(
        type='OLNKMeansFFSRoIHead',
        start_epoch=12,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
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
dataset_type = "VOSCocoSplitDataset"


data_root = 'data/voc0712/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsWithAnnID', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                   (736, 1333), (768, 1333), (800, 1333)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PseudoLabelFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_ann_ids', 'gt_pseudo_labels',
                               'gt_weak_bboxes', 'gt_weak_bboxes_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='voc',
        type=dataset_type,
        pipeline=train_pipeline,
        ann_file=data_root + 'voc0712_train_all.json',
        img_prefix=data_root + 'JPEGImages/',
        ),
    val=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='voc',
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file=data_root + 'val_coco_format.json',
        img_prefix=data_root + 'JPEGImages/'
    ),
    test=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='voc',
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file=data_root + 'val_coco_format.json',
        img_prefix=data_root + 'JPEGImages/'
    ))

custom_hooks = [dict(type='SetEpochInfoHook')]
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[4])
total_epochs = 8

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

load_from = './work_dirs/oln_mask/epoch_8.pth'