dataset_type = 'BDDSplitDataset'
data_root = 'data/bdd100k/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
        type=dataset_type,
        ann_file=data_root + 'train_bdd_converted.json',
        img_prefix=data_root + 'bdd100k/images/100k/train',
        pipeline=train_pipeline,
        is_class_agnostic=True,
        train_class='all',
        eval_class='all',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_bdd_converted.json',
        img_prefix=data_root + 'bdd100k/images/100k/val',
        pipeline=test_pipeline,
        is_class_agnostic=True,
        train_class='all',
        eval_class='all'
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_bdd_converted.json',
        img_prefix=data_root + 'bdd100k/images/100k/val',
        pipeline=test_pipeline,
        is_class_agnostic=True,
        train_class='all',
        eval_class='all',
    ))
evaluation = dict(interval=1, metric='bbox')
