_base_ = './oln_box.py'

# Dataset
dataset_type = 'CocoSplitDataset'
data_root = 'data/cityscapes/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
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
        img_scale=(2048, 1024),
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
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            is_class_agnostic=True,
            ann_file=data_root +
                     'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        is_class_agnostic=True,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        is_class_agnostic=True,
        ann_file=data_root +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
total_epochs = 8

checkpoint_config = dict(interval=2)
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
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
workflow = [('train', 1)]

work_dir = './work_dirs/oln_box_cityscapes/'
