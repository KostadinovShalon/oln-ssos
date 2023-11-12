dataset_type = 'DB6SplitDataset'
data_root = 'data/db6/'

custom_imports = dict(
    imports=[
        'vos.datasets.vos_coco',
        'vos.datasets.pipelines.loading'
    ],
    allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsWithAnnID', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_ann_ids', 'gt_pseudo_labels']),
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
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_coco_no_firearms.json',
        img_prefix=data_root + 'images/',
        is_class_agnostic=True,
        train_class='id',
        eval_class='id',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test_coco_no_firearms.json',
        img_prefix=data_root + 'images/',
        is_class_agnostic=True,
        train_class='id',
        eval_class='id',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test_coco_no_firearms.json',
        img_prefix=data_root + 'images/',
        is_class_agnostic=True,
        train_class='id',
        eval_class='id',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
