_base_ = [
    './oln_box_model.py',
    '../_base_/datasets/coco_split_detection_ann_id.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

data_root = 'data/voc0712/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
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
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        is_class_agnostic=True,
        train_class='all',
        eval_class='all',
        pipeline=train_pipeline,
        ann_file=data_root + 'voc0712_train_all.json',
        img_prefix=data_root + 'JPEGImages/',
        ),
    val=dict(
        is_class_agnostic=True,
        train_class='all',
        eval_class='all',
        pipeline=test_pipeline,
        ann_file=data_root + 'val_coco_format.json',
        img_prefix=data_root + 'JPEGImages/'
    ),
    test=dict(
        is_class_agnostic=True,
        train_class='all',
        eval_class='all',
        pipeline=test_pipeline,
        ann_file=data_root + 'val_coco_format.json',
        img_prefix=data_root + 'JPEGImages/'
    ))


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6, 7])
total_epochs = 8

checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
