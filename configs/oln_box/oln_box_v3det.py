base = './oln_box.py'

model = dict(
    # model training and testing settings
    train_cfg=dict(
        rpn_proposal=dict(nms_pre=4000),
        rcnn=dict(
            assigner=dict(
                perm_repeat_gt_cfg=dict(iou_thr=0.7)))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=500)))

data_root = 'data/V3Det/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        ann_file=data_root + 'annotations/v3det_2023_v1_train.json',
        img_prefix=data_root + 'images/',
        train_class='all',
        eval_class='all',
        pipeline=train_pipeline,
        ),
    val=dict(
        ann_file=data_root + 'annotations/v3det_2023_v1_val.json',
        img_prefix=data_root + 'images/',),
    test=dict(
        ann_file=data_root + 'annotations/v3det_2023_v1_val.json',
        img_prefix=data_root + 'images/',))

# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 2048,
    step=[6, 7])
total_epochs = 8
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
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

work_dir='./work_dirs/oln_box_v3det/'