_base_ = '../oln_mask/oln_mask_cityscapes.py'

custom_imports = dict(
    imports=[
        'vos.models.roi_heads.oln_mask_vos_roi_head',
        'vos.models.detectors.epoch_faster_rcnn',
        'vos.models.roi_heads.bbox_heads.oln_vos_bbox_head',
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

model = dict(
    type='EpochMaskRCNN',
    roi_head=dict(
        type='OLNMaskKMeansVOSRoIHead',
        start_epoch=4,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        pseudo_label_loss_weight=1.,
        k=5,
        repeat_ood_sampling=4,
        bbox_head=dict(
            type='VOSShared2FCBBoxScoreHead',
        reg_class_agnostic=True))
    )

checkpoint_config = dict(interval=1)
dataset_type = "VOSCocoSplitDataset"

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[8])
total_epochs = 10

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        dataset=dict(type=dataset_type, pipeline=train_pipeline)),
    val=dict(
        type=dataset_type),
    test=dict(
        type=dataset_type)
)

custom_hooks = [dict(type='SetEpochInfoHook')]
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# Pretrained on COCO
load_from = './work_dirs/oln_mask/epoch_8.pth'
work_dir = './work_dirs/oln_vos_mask_cityscapes_kmeans_5/'
