_base_ = './oln_box.py'

# Dataset
dataset_type = 'OI7CocoSplitDataset'
data_root = 'data/oi7/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 800), keep_ratio=True),
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
        img_scale=(1024, 800),
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
    samples_per_gpu=56,
    workers_per_gpu=2,
    train=dict(
        is_class_agnostic=True,
        train_class='all',
        eval_class='all',
        type=dataset_type,
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline,
        ann_file=data_root + 'annotations/detection_train_curated.json',
    ),
    val=dict(
        is_class_agnostic=True,
        type=dataset_type,
        train_class='all',
        eval_class='all',
        ann_file=data_root + 'annotations/detection_validation.json',
        img_prefix=data_root + 'images/validation/',
        pipeline=test_pipeline),
    test=dict(
        is_class_agnostic=True,
        type=dataset_type,
        train_class='all',
        eval_class='all',
        ann_file=data_root + 'annotations/detection_test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.07, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=26000,
    warmup_ratio=1.0 / 64,
    step=[7])
total_epochs = 8

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]

work_dir = './work_dirs/oln_box_oi7/'
