_base_ = [
    '../faster_rcnn/faster-rcnn_r50_fpn_1x_voc0712.py'
]

custom_imports = dict(
    imports=[
        'vos.models.detectors.epoch_faster_rcnn',
        'vos.models.roi_heads.vos_roi_head',
        'vos.models.roi_heads.bbox_heads.vos_convfc_bbox_head',
        'vos.datasets.vos_coco'
    ],
    allow_failed_imports=False)
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
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

model = dict(
    type='EpochFasterRCNN',
    roi_head=dict(
        type='VOSRoIHead',
        vos_samples_per_class=1000,
        start_epoch=12,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        bbox_head=dict(
            type='VOSConvFCBBoxHead'))
    )

dataset_type = "VOSCocoDataset"
data = dict(
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type),
    test=dict(
        type=dataset_type))
custom_hooks = [dict(type='SetEpochInfoHook')]
resume_from = "work_dirs/vos_faster-rcnn_r50_fpn_1x_voc0712/epoch_17.pth"
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 16])
total_epochs = 18
